import modal
from fastapi import Request
from fastapi.responses import Response

app = modal.App("ddpo")

llava_weights_vol = modal.Volume.from_name("llava-weights")
wandb_vol = modal.Volume.from_name("wandb-logs")

llava_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "transformers>=4.57,<5.0.0",
        "accelerate",
        "torch",
        "torchvision",
        "flask",
        "gunicorn",
        "pillow",
        "numpy",
        "bert-score",
        "pydantic<2",
        "fastapi[standard]",
        force_build=True
    )
    .add_local_dir(
        "~/cs231n/LLaVA-server", remote_path="/root/cs231n/LLaVA-server", copy=True
    )
    .run_commands(
        "cd /root/cs231n/LLaVA-server/LLaVA && pip install -e . --no-deps"
    )
)

train_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "torchvision",
        "transformers",
        "accelerate",
        "diffusers==0.17.1",
        "huggingface_hub==0.22.2",
        "wandb",
        "absl-py",
        "ml_collections",
        "inflect",
        "pydantic<2",
        "numpy",
        "pillow",
        "tqdm",
        "requests",
        "fastapi[standard]",
    )
    .add_local_dir(
        "~/cs231n/ddpo-pytorch", remote_path="/root/cs231n/ddpo-pytorch", copy=True
    )
    .run_commands("cd /root/cs231n/ddpo-pytorch && pip install -e .")
)


@app.cls(
    gpu="L40S",
    image=llava_image,
    volumes={"/llava-weights": llava_weights_vol},
    scaledown_window=300,
    min_containers=0,
    max_containers=2,
    timeout=600,
)
@modal.concurrent(target_inputs=3, max_inputs=6)
class LLaVAServer:
    @modal.enter()
    def load_model(self):
        import sys

        sys.path.append("/root/cs231n/LLaVA-server")
        from llava_server.bertscore import load_bertscore
        from llava_server.llava import load_llava

        self.inference_fn = load_llava("/llava-weights/llava-weights")
        self.bertscore_fn = load_bertscore()

    @modal.fastapi_endpoint(method="POST")
    async def inference(self, request: Request) -> Response:
        import pickle
        import traceback
        from io import BytesIO

        import numpy as np
        from PIL import Image

        print(f"received POST request from {request.client.host}")
        data = await request.body()

        try:
            data = pickle.loads(data)

            images = [Image.open(BytesIO(d), formats=["jpeg"]) for d in data["images"]]
            queries = data["queries"]

            print(f"Got {len(images)} images, {len(queries[0])} queries per image")

            outputs = self.inference_fn(images, queries)

            response = {"outputs": outputs}

            if "answers" in data:
                print(f"Running bertscore...")
                output_shape = np.array(outputs).shape
                (
                    response["precision"],
                    response["recall"],
                    response["f1"],
                ) = self.bertscore_fn(
                    np.array(outputs).reshape(-1).tolist(),
                    np.array(data["answers"]).reshape(-1).tolist(),
                )

                for key in ["precision", "recall", "f1"]:
                    response[key] = response[key].reshape(output_shape).tolist()

            response = pickle.dumps(response)
            returncode = 200
            return Response(
                content=response,
                status_code=returncode,
                media_type="application/octet-stream",
            )
        except Exception as e:
            response = traceback.format_exc()
            print(response)
            response = response.encode("utf-8")
            returncode = 500

            return Response(
                content=response,
                status_code=returncode,
                media_type="text/plain",
            )
@app.cls(
    gpu="L40S",
    image=llava_image,
    volumes={"/llava-weights": llava_weights_vol},
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
    ],
    scaledown_window=300,
    min_containers=0,
    max_containers=2,
    timeout=600
)
@modal.concurrent(target_inputs=3, max_inputs=6)
class QwenServer:
    @modal.enter()
    def load_model(self):
        import sys

        sys.path.append("/root/cs231n/LLaVA-server")
        from llava_server.bertscore import load_bertscore
        from llava_server.llava import load_qwen3vl, load_llava

        self.inference_fn = load_qwen3vl("internlm/Spatial-SSRL-Qwen3VL-4B")
        self.bertscore_fn = load_bertscore()

    @modal.fastapi_endpoint(method="POST")
    async def inference(self, request: Request) -> Response:
        import pickle
        import traceback
        from io import BytesIO

        import numpy as np
        from PIL import Image

        print(f"received POST request from {request.client.host}")
        data = await request.body()

        try:
            data = pickle.loads(data)

            images = [Image.open(BytesIO(d), formats=["jpeg"]) for d in data["images"]]
            queries = data["queries"]

            print(f"Got {len(images)} images, {len(queries[0])} queries per image")

            outputs = self.inference_fn(images, queries)

            response = {"outputs": outputs}

            if "answers" in data:
                print(f"Running bertscore...")
                output_shape = np.array(outputs).shape
                (
                    response["precision"],
                    response["recall"],
                    response["f1"],
                ) = self.bertscore_fn(
                    np.array(outputs).reshape(-1).tolist(),
                    np.array(data["answers"]).reshape(-1).tolist(),
                )

                for key in ["precision", "recall", "f1"]:
                    response[key] = response[key].reshape(output_shape).tolist()

            response = pickle.dumps(response)
            returncode = 200

            return Response(
                content=response,
                status_code=returncode,
                media_type="application/octet-stream",
            )
        except Exception as e:
            response = traceback.format_exc()
            print(response)
            response = response.encode("utf-8")
            returncode = 500

            return Response(
                content=response,
                status_code=returncode,
                media_type="text/plain",
            )


@app.function(
    gpu="H100:4",
    image=train_image,
    timeout=60 * 60 * 8,
    secrets=[
        modal.Secret.from_name("wandb"),
        modal.Secret.from_name("huggingface-secret"),
    ],
    volumes={"/wandb-logs": wandb_vol},
)
def train():
    import os
    import subprocess

    os.environ["WANDB_DIR"] = "/wandb-logs/cs231n"

    server = LLaVAServer()
    url = "https://tylerho--ddpo-llavaserver-inference-dev.modal.run"
    # qwen_url = server.inference.web_url
    # server = QwenServer()
    # url = "https://tylerho--ddpo-qwenserver-inference-dev.modal.run"
    env = os.environ.copy()
    env["LLAVA_SERVER_URL"] = url
    subprocess.run(
        [
            "accelerate",
            "launch",
            "--num_processes",
            "4",
            "--multi_gpu",
            "scripts/train.py",
        ],
        cwd="/root/cs231n/ddpo-pytorch",
        env=env,
        check=True,
    )


@app.local_entrypoint()
def main():
    train.remote()
