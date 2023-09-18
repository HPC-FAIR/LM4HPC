from .pipeline_openmp_question_answering import openmp_question_answering
# from .pipeline_codebase_question_answering import codebase_question_answering
from .pipeline_similarity_checking import similarity_checking

class Pipeline:
    def __init__(self, model: str, **kwargs):
        self.model = model
        self.parameters = kwargs

    def run(self, *args, **kwargs):
        raise NotImplementedError("Subclasses should implement this method!")

    def preprocess(self, *args, **kwargs):
        """Any preprocessing logic common to all pipelines can go here."""
        return args, kwargs

    def postprocess(self, result):
        """Any postprocessing logic common to all pipelines can go here."""
        return result
    
class OpenMPQuestionAnsweringPipeline(Pipeline):
    def run(self, question):
        # preprocessed_question, _ = self.preprocess(question)
        answer = openmp_question_answering(self.model, question, **self.parameters)
        return self.postprocess(answer)
    
# class CodebaseQuestionAnsweringPipeline(Pipeline):
#     def run(self, question):
#         preprocessed_question, _ = self.preprocess(question)
#         answer = codebase_question_answering(self.model, preprocessed_question, **self.parameters)
#         return self.postprocess(answer)
    
# class SimilarityCheckingPipeline(Pipeline):
#     def run(self, code1, code2):
#         preprocessed_code1, preprocessed_code2 = self.preprocess(code1, code2)
#         similarity_score = similarity_checking(self.model, preprocessed_code1, preprocessed_code2, **self.parameters)
#         return self.postprocess(similarity_score)