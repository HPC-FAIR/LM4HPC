from .pipeline_openmp_question_answering import openmp_question_answering


def hpcpipelines(task, model, **kwargs):
    if task == 'openmp_question_answering':
        return lambda question: openmp_question_answering(model, question)
    # elif task == 'similarity_checking':
    #     return similarity_checking(model, **kwargs)
    # elif task == 'codebase_question_answering':
    #     return codebase_question_answering(model, **kwargs)
    else:
        raise ValueError('Unknown task: {}'.format(task))