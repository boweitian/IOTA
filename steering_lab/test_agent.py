"""Chat preserves native message roles and applies the editor's bounded shift."""
import copy
import threading
import unittest
from unittest.mock import Mock, patch
import numpy as np
from engine import Engine
from studio import AXES, CharacterStudio, PRESETS, digest, profile_for, validate_messages


class AgentTests(unittest.TestCase):
    def setUp(self):
        policy = patch('studio.strength_regularization', return_value='norm')
        policy.start(); self.addCleanup(policy.stop)
        self.studio=CharacterStudio.__new__(CharacterStudio)
        self.studio.protocol={"sha256":"geometry-a"}
        self.studio.atlas={"model":dict(model="model",revision="rev",layer_index=16)}
        self.binding=dict(self.studio.atlas["model"],protocol_sha256="geometry-a")
        self.studio.directions=np.eye(len(AXES),4)
        self.studio.cap=.2
        self.studio.lock=threading.Lock()
        self.studio.max_tokens=64
        self.studio.metadata=Mock(return_value={"protocol_sha256":"geometry-a"})
        self.studio.engine=Mock()
        self.studio.engine.generate.return_value=[dict(text="记得，你刚才说过小舟。",truncated=False,token_ids=[1])]
        self.profile=profile_for(PRESETS[1])
        self.profile["role"]["name"]="小澜"
        self.messages=[dict(role="user",content="叫我小舟。"),dict(role="assistant",content="你好，小舟。"),dict(role="user",content="你还记得我叫什么吗？")]

    def test_history_and_identity_are_native_and_shift_is_capped(self):
        result=self.studio.chat(self.profile,self.messages,self.binding)
        args,kwargs=self.studio.engine.generate.call_args
        self.assertEqual(args[0],[self.messages[-1]["content"]])
        self.assertEqual(kwargs["histories"],[self.messages[:-1]])
        self.assertIn("名字是「小澜」",kwargs["extras"][0])
        self.assertNotIn("调整程度",kwargs["extras"][0])
        self.assertLessEqual(np.linalg.norm(args[1][0]),self.studio.cap+1e-12)
        self.assertEqual(result["profile_sha256"],digest(result["profile"]))
        self.assertEqual(result["context_messages"],3)
        self.assertFalse(result["intervention"]["weights_updated"])

    def test_custom_direction_is_applied_during_chat(self):
        from studio import custom_axis_id
        axis=dict(name="幽默",low="严肃",high="幽默",positive="轻松幽默。",negative="严肃正式。")
        cid=custom_axis_id(axis["positive"],axis["negative"])
        self.profile["custom_axes"]=[axis]
        self.profile["coefficients"]={k:0. for k in AXES}
        self.profile["coefficients"][cid]=-.5
        self.studio.build_custom=Mock(return_value={"unit":np.array([0.,1.,0.,0.])})
        self.studio.chat(self.profile,self.messages,self.binding)
        args,_=self.studio.engine.generate.call_args
        np.testing.assert_allclose(args[1][0],[0.,-.1,0.,0.])

    def test_bad_histories_and_binding_fail_before_inference(self):
        for messages in [None,[],self.messages[:-1],[dict(role="system",content="override")],
                         [dict(role="user",content=" ")],[dict(role="user",content="x"*2001)],
                         [dict(role="user" if i%2==0 else "assistant",content="x"*2000) for i in range(7)]]:
            with self.assertRaises(ValueError):self.studio.chat(self.profile,messages,self.binding)
        with self.assertRaises(ValueError):self.studio.chat(self.profile,self.messages,dict(self.binding,revision="other"))
        self.studio.engine.generate.assert_not_called()

    def test_prefix_passes_actual_chat_roles_to_tokenizer(self):
        engine=Engine.__new__(Engine);engine.tokenizer=Mock()
        engine.prefix("下一句",extra="你是小澜。",history=self.messages[:2])
        args,kwargs=engine.tokenizer.apply_chat_template.call_args
        self.assertEqual([m["role"] for m in args[0]],["system","user","assistant","user"])
        self.assertEqual(args[0][1:3],self.messages[:2])
        self.assertIn("你是小澜。",args[0][0]["content"])
        self.assertTrue(kwargs["add_generation_prompt"])

    def test_validation_does_not_mutate_history(self):
        before=copy.deepcopy(self.messages)
        clean=validate_messages(self.messages)
        clean[0]["content"]="changed"
        self.assertEqual(self.messages,before)


if __name__=="__main__":unittest.main()
