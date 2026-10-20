"""Registered response directions retain their extraction method, binding, and injection positions."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import direction_library
from studio import validate_custom_axes

class DirectionLibraryTests(unittest.TestCase):
    def setUp(self):
        directory=tempfile.TemporaryDirectory();self.addCleanup(directory.cleanup)
        self.directory=Path(directory.name)
        patcher=patch('direction_library.DIRECTORY',self.directory);patcher.start();self.addCleanup(patcher.stop)
        self.spec=dict(name='测试方向',low='负向',high='正向',positive='正向回答。',negative='负向回答。',method='response_mean',positions='all')
        self.axis=validate_custom_axes([self.spec])[0]
        self.atlas=dict(model=dict(model='model',revision='rev',layer_index=16,hidden_size=4),scale=1.)
        self.payload=dict(axis=self.axis,native_direction=[1.,0.,0.,0.],positions='all',
            provenance=dict(model=self.atlas['model'],protocol_sha256='protocol'))
        self.path=self.directory/(self.axis['id']+'.json');self.write()

    def write(self):self.path.write_text(json.dumps(self.payload))

    def test_registered_direction_and_method_survive_recipe_validation(self):
        self.assertEqual(validate_custom_axes([self.axis])[0],self.axis)
        loaded=direction_library.registered_direction(self.axis,self.atlas,'protocol')
        np.testing.assert_allclose(loaded['native_direction'],[1.,0.,0.,0.])
        self.assertEqual(direction_library.available_directions(self.atlas,'protocol'),[self.payload])
        default=validate_custom_axes([{k:v for k,v in self.spec.items() if k not in ['method','positions']}])[0]
        last=validate_custom_axes([dict(self.spec,positions='last')])[0]
        self.assertEqual(len({default['id'],last['id'],self.axis['id']}),3)

    def test_wrong_geometry_or_model_is_rejected(self):
        with self.assertRaises(ValueError):direction_library.registered_direction(self.axis,self.atlas,'other')
        for key,value in [('model','other'),('revision','other'),('layer_index',20),('hidden_size',8)]:
            wrong=copy.deepcopy(self.atlas);wrong['model'][key]=value
            with self.assertRaises(ValueError):direction_library.registered_direction(self.axis,wrong,'protocol')

    def test_missing_asset_or_changed_vector_cannot_fall_back_to_intent(self):
        self.path.unlink()
        with self.assertRaises(ValueError):direction_library.registered_direction(self.axis,self.atlas,'protocol')
        for vector in [[0.,0.,0.,0.],[2.,0.,0.,0.],[1.],[float('nan'),0.,0.,0.]]:
            self.payload['native_direction']=vector;self.write()
            with self.assertRaises(ValueError):direction_library.registered_direction(self.axis,self.atlas,'protocol')

    def test_injection_position_cannot_be_changed_without_changing_identity(self):
        with self.assertRaises(ValueError):validate_custom_axes([dict(self.axis,positions='last')])
        with self.assertRaises(ValueError):validate_custom_axes([dict(self.spec,method='unknown')])
        self.payload['positions']='last';self.write()
        with self.assertRaises(ValueError):direction_library.registered_direction(self.axis,self.atlas,'protocol')

if __name__=='__main__':unittest.main()
