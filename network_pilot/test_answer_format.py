import unittest
from answer_format import analysis_prompt,parse_answer

class AnswerFormatTests(unittest.TestCase):
    def test_scores_final_line_not_letters_in_analysis(self):
        value=parse_answer('A看似合理，但证据更支持C。\n答案：C',list('ABC'))
        self.assertEqual(value['prediction'],2)
    def test_generic_label_sets(self):
        for letters,answer in [('AB','B'),('ABCDE','E')]:
            self.assertEqual(parse_answer('简短依据。\n答案：'+answer,list(letters))['prediction'],letters.index(answer))
    def test_ascii_colon_and_trailing_whitespace(self):
        self.assertTrue(parse_answer('依据。\n答案: B  \n',list('ABC'))['format_valid'])
    def test_invalid_or_ambiguous_output_is_not_guessed(self):
        for text in ['A','依据中提到A','依据。\nFinal: A','答案：A','依据。\n答案：A/B/C','依据。\n答案：A\n答案：B','依据。\n答案：Z','依据。\n答案：A\n补充文字']:
            with self.subTest(text=text):self.assertFalse(parse_answer(text,list('ABC'))['format_valid'])
    def test_truncated_answer_is_invalid(self):
        self.assertFalse(parse_answer('依据。\n答案：A',list('ABC'),truncated=True)['format_valid'])
    def test_format_instruction_changes_only_output_request(self):
        before='Definitions stay fixed.\nRespond with exactly one letter: A, B, or C.\nPremise: {context}\nHypothesis: {hypothesis}'
        after=analysis_prompt(before,list('ABC'))
        self.assertTrue(after.startswith('Definitions stay fixed.\n'))
        self.assertTrue(after.endswith('Premise: {context}\nHypothesis: {hypothesis}'))
        self.assertIn('答案：',after)

if __name__=='__main__':unittest.main()
