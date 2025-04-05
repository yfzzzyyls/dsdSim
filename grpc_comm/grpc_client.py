import grpc
import inference_pb2_grpc

def create_stub(target_address):
    """
    Create and return a gRPC client stub for the SpeculativeService at the given address.
    """
    channel = grpc.insecure_channel(target_address)
    stub = inference_pb2_grpc.SpeculativeServiceStub(channel)
    # We could add connectivity check here
    return stub