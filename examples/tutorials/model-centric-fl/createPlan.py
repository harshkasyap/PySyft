import syft as sy
from syft.serde import protobuf
from syft_proto.execution.v1.plan_pb2 import Plan as PlanPB
from syft_proto.execution.v1.state_pb2 import State as StatePB
from syft.grid.clients.model_centric_fl_client import ModelCentricFLClient
from syft.execution.state import State
from syft.execution.placeholder import PlaceHolder
from syft.execution.translation import TranslationTarget

import torch as th
from torch import nn

import os
import websockets
import json
import requests

sy.make_hook(globals())
# force protobuf serialization for tensors
hook.local_worker.framework = None
th.random.manual_seed(1)

def set_model_params(module, params_list, start_param_idx=0):
    """ Set params list into model recursively
    """
    param_idx = start_param_idx

    for name, param in module._parameters.items():
        module._parameters[name] = params_list[param_idx]
        param_idx += 1

    for name, child in module._modules.items():
        if child is not None:
            param_idx = set_model_params(child, params_list, param_idx)

    return param_idx

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 392)
        self.fc2 = nn.Linear(392, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        return x

model = Net()

def softmax_cross_entropy_with_logits(logits, targets, batch_size):
    """ Calculates softmax entropy
        Args:
            * logits: (NxC) outputs of dense layer
            * targets: (NxC) one-hot encoded labels
            * batch_size: value of N, temporarily required because Plan cannot trace .shape
    """
    # numstable logsoftmax
    norm_logits = logits - logits.max()
    log_probs = norm_logits - norm_logits.exp().sum(dim=1, keepdim=True).log()
    # NLL, reduction = mean
    return -(targets * log_probs).sum() / batch_size

def naive_sgd(param, **kwargs):
    return param - kwargs['lr'] * param.grad


@sy.func2plan()
def training_plan(X, y, batch_size, lr, model_params):
    # inject params into model
    set_model_params(model, model_params)

    # forward pass
    logits = model.forward(X)
    
    # loss
    loss = softmax_cross_entropy_with_logits(logits, y, batch_size)

    # backprop
    loss.backward()

    # step
    updated_params = [
        naive_sgd(param, lr=lr)
        for param in model_params
    ]
    
    # accuracy
    pred = th.argmax(logits, dim=1)
    target = th.argmax(y, dim=1)
    acc = pred.eq(target).sum().float() / batch_size

    return (
        loss,
        acc,
        *updated_params
    )

# Dummy input parameters to make the trace
model_params = [param.data for param in model.parameters()]  # raw tensors instead of nn.Parameter
X = th.randn(3, 28 * 28)
y = nn.functional.one_hot(th.tensor([1, 2, 3]), 10)
lr = th.tensor([0.01])
batch_size = th.tensor([3.0])

_ = training_plan.build(X, y, batch_size, lr, model_params, trace_autograd=True)

print(training_plan.code)
print(training_plan.torchscript.code)

training_plan.base_framework = TranslationTarget.TENSORFLOW_JS.value
print(training_plan.code)
training_plan.base_framework = TranslationTarget.PYTORCH.value

@sy.func2plan()
def avg_plan(avg, item, num):
    new_avg = []
    for i, param in enumerate(avg):
        new_avg.append((avg[i] * num + item[i]) / (num + 1))
    return new_avg

# Build the Plan
_ = avg_plan.build(model_params, model_params, th.tensor([1.0]))

# Let's check Plan contents
print(avg_plan.code)

# Test averaging plan
# Pretend there're diffs, all params of which are ones * dummy_coeffs
dummy_coeffs = [1, 5.5, 7, 55]
dummy_diffs = [[th.ones_like(param) * i for param in model_params] for i in dummy_coeffs]
mean_coeff = th.tensor(dummy_coeffs).mean().item()

# Remove original function to make sure we execute traced Plan
avg_plan.forward = None

# Calculate avg value using our plan
avg = dummy_diffs[0]
for i, diff in enumerate(dummy_diffs[1:]):
    avg = avg_plan(list(avg), diff, th.tensor([i + 1]))

# Avg should be ones*mean_coeff for each param
for i, param in enumerate(model_params):
    expected = th.ones_like(param) * mean_coeff
    assert avg[i].eq(expected).all(), f"param #{i}"

# PyGrid Node address
gridAddress = "alice:5000"
grid = ModelCentricFLClient(id="test", address=gridAddress, secure=False)
grid.connect()# These name/version you use in worker
name = "mnist"
version = "1.0.0"

client_config = {
    "name": name,
    "version": version,
    "batch_size": 64,
    "lr": 0.005,
    "max_updates": 100  # custom syft.js option that limits number of training loops per worker
}

server_config = {
    "min_workers": 5,
    "max_workers": 5,
    "pool_selection": "random",
    "do_not_reuse_workers_until_cycle": 6,
    "cycle_length": 28800,  # max cycle length in seconds
    "num_cycles": 5,  # max number of cycles
    "max_diffs": 1,  # number of diffs to collect before avg
    "minimum_upload_speed": 0,
    "minimum_download_speed": 0,
    "iterative_plan": True  # tells PyGrid that avg plan is executed per diff
}

private_key = """
-----BEGIN RSA PRIVATE KEY-----
MIIEowIBAAKCAQEAzQMcI09qonB9OZT20X3Z/oigSmybR2xfBQ1YJ1oSjQ3YgV+G
FUuhEsGDgqt0rok9BreT4toHqniFixddncTHg7EJzU79KZelk2m9I2sEsKUqEsEF
lMpkk9qkPHhJB5AQoClOijee7UNOF4yu3HYvGFphwwh4TNJXxkCg69/RsvPBIPi2
9vXFQzFE7cbN6jSxiCtVrpt/w06jJUsEYgNVQhUFABDyWN4h/67M1eArGA540vyd
kYdSIEQdknKHjPW62n4dvqDWxtnK0HyChsB+LzmjEnjTJqUzr7kM9Rzq3BY01DNi
TVcB2G8t/jICL+TegMGU08ANMKiDfSMGtpz3ZQIDAQABAoIBAD+xbKeHv+BxxGYE
Yt5ZFEYhGnOk5GU/RRIjwDSRplvOZmpjTBwHoCZcmsgZDqo/FwekNzzuch1DTnIV
M0+V2EqQ0TPJC5xFcfqnikybrhxXZAfpkhtU+gR5lDb5Q+8mkhPAYZdNioG6PGPS
oGz8BsuxINhgJEfxvbVpVNWTdun6hLOAMZaH3DHgi0uyTBg8ofARoZP5RIbHwW+D
p+5vd9x/x7tByu76nd2UbMp3yqomlB5jQktqyilexCIknEnfb3i/9jqFv8qVE5P6
e3jdYoJY+FoomWhqEvtfPpmUFTY5lx4EERCb1qhWG3a7sVBqTwO6jJJBsxy3RLIS
Ic0qZcECgYEA6GsBP11a2T4InZ7cixd5qwSeznOFCzfDVvVNI8KUw+n4DOPndpao
TUskWOpoV8MyiEGdQHgmTOgGaCXN7bC0ERembK0J64FI3TdKKg0v5nKa7xHb7Qcv
t9ccrDZVn4y/Yk5PCqjNWTR3/wDR88XouzIGaWkGlili5IJqdLEvPvUCgYEA4dA+
5MNEQmNFezyWs//FS6G3lTRWgjlWg2E6BXXvkEag6G5SBD31v3q9JIjs+sYdOmwj
kfkQrxEtbs173xgYWzcDG1FI796LTlJ/YzuoKZml8vEF3T8C4Bkbl6qj9DZljb2j
ehjTv5jA256sSUEqOa/mtNFUbFlBjgOZh3TCsLECgYAc701tdRLdXuK1tNRiIJ8O
Enou26Thm6SfC9T5sbzRkyxFdo4XbnQvgz5YL36kBnIhEoIgR5UFGBHMH4C+qbQR
OK+IchZ9ElBe8gYyrAedmgD96GxH2xAuxAIW0oDgZyZgd71RZ2iBRY322kRJJAdw
Xq77qo6eXTKpni7grjpijQKBgDHWRAs5DVeZkTwhoyEW0fRfPKUxZ+ZVwUI9sxCB
dt3guKKTtoY5JoOcEyJ9FdBC6TB7rV4KGiSJJf3OXAhgyP9YpNbimbZW52fhzTuZ
bwO/ZWC40RKDVZ8f63cNsiGz37XopKvNzu36SJYv7tY8C5WvvLsrd/ZxvIYbRUcf
/dgBAoGBAMdR5DXBcOWk3+KyEHXw2qwWcGXyzxtca5SRNLPR2uXvrBYXbhFB/PVj
h3rGBsiZbnIvSnSIE+8fFe6MshTl2Qxzw+F2WV3OhhZLLtBnN5qqeSe9PdHLHm49
XDce6NV2D1mQLBe8648OI5CScQENuRGxF2/h9igeR4oRRsM1gzJN
-----END RSA PRIVATE KEY-----
""".strip()

public_key = """
-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAzQMcI09qonB9OZT20X3Z
/oigSmybR2xfBQ1YJ1oSjQ3YgV+GFUuhEsGDgqt0rok9BreT4toHqniFixddncTH
g7EJzU79KZelk2m9I2sEsKUqEsEFlMpkk9qkPHhJB5AQoClOijee7UNOF4yu3HYv
GFphwwh4TNJXxkCg69/RsvPBIPi29vXFQzFE7cbN6jSxiCtVrpt/w06jJUsEYgNV
QhUFABDyWN4h/67M1eArGA540vydkYdSIEQdknKHjPW62n4dvqDWxtnK0HyChsB+
LzmjEnjTJqUzr7kM9Rzq3BY01DNiTVcB2G8t/jICL+TegMGU08ANMKiDfSMGtpz3
ZQIDAQAB
-----END PUBLIC KEY-----
""".strip()

server_config["authentication"] = {
    "type": "jwt",
    "pub_key": public_key,
}

model_params_state = State(
    state_placeholders=[
        PlaceHolder().instantiate(param)
        for param in model_params
    ]
)

response = grid.host_federated_training(
    model=model_params_state,
    client_plans={'training_plan': training_plan},
    client_protocols={},
    server_averaging_plan=avg_plan,
    client_config=client_config,
    server_config=server_config
)

print("Host response:", response)