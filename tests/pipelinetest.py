from lm4hpc import hpcpipeline

def test_pipeline_functionality():
    # Here, call functions from hpcpipelne and assert expected behaviors.
    pipeline = hpcpipeline.get_pipeline('openmp_question_answering', 'gpt-3.5-turb')
    result = pipeline.run('what is openmp?')
    assert result == 'expected_output', f"Expected 'expected_output' but got {result}"
