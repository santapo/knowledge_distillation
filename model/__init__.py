from student import StudentModel
from teacher import TeacherModel

_model_factory_ = {
    'student': StudentModel,
    'teacher': TeacherModel
}


def get_model(model_name: str):
    try:
        model = _model_factory_[model_name]()
    except:
        print(f'{model_name} is not defined')
        return
    return model