import grpc
import inference_pb2_grpc
import management_pb2
import management_pb2_grpc


def get_inference_stub():
    channel = grpc.insecure_channel('localhost:7070')
    return inference_pb2_grpc.InferenceAPIsServiceStub(channel)


def get_management_stub():
    channel = grpc.insecure_channel('localhost:7071')
    return management_pb2_grpc.ManagementAPIsServiceStub(channel)


def run_management_api(api_name, **kwargs):
    management_stub = get_management_stub()
    return getattr(management_stub, api_name)(getattr(management_pb2, f"{api_name}Request")(**kwargs))

