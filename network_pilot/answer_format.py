"""Shared generation-first answer protocol for arbitrary MCQ label sets."""
import re

def analysis_prompt(template,letters):
    instruction=('先用中文给出1至2句简短分析，说明判断依据。最后单独一行输出一个选项，'
        '格式为“答案：<选项字母>”，可选字母为'+ '/'.join(letters)+'。例如“答案：'+letters[0]+'”。'
        '最后一行不要输出斜杠、尖括号或其他文字。')
    pattern=r'Respond with exactly one letter:[^\n]*'
    if not re.search(pattern,template):raise ValueError('Missing original output instruction')
    return re.sub(pattern,lambda _:instruction,template)

def parse_answer(text,letters,truncated=False):
    if truncated:return dict(prediction=None,format_valid=False,error='truncated')
    lines=text.strip().splitlines()
    if not lines:return dict(prediction=None,format_valid=False,error='empty')
    matches=re.findall(r'^\s*答案[：:]\s*([A-Z])\s*$',text,re.M)
    final=re.fullmatch(r'\s*答案[：:]\s*([A-Z])\s*',lines[-1])
    if len(matches)>1:return dict(prediction=None,format_valid=False,error='multiple_answer_lines')
    if not final:return dict(prediction=None,format_valid=False,error='missing_final_answer_line')
    if final[1] not in letters:return dict(prediction=None,format_valid=False,error='invalid_option')
    if not '\n'.join(lines[:-1]).strip():return dict(prediction=None,format_valid=False,error='missing_analysis')
    return dict(prediction=letters.index(final[1]),format_valid=True,error=None)
