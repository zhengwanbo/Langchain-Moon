from fastchat.conversation import Conversation
from server.model_workers.base import *
from fastchat import conversation as conv
import sys
from typing import List, Dict, Iterator, Literal
from configs import logger, log_verbose
import requests
import jwt
import time
import json

class XinferWorker(ApiModelWorker):
    def __init__(
            self,
            *,
            model_names: List[str] = ["xinfer"],
            controller_addr: str = None,
            worker_addr: str = None,
            version: Literal["xinfer"] = "xinfer",
            **kwargs,
    ):
        kwargs.update(model_names=model_names, controller_addr=controller_addr, worker_addr=worker_addr)
        kwargs.setdefault("context_len", 4096)
        super().__init__(**kwargs)
        self.version = version

    def do_chat(self, params: ApiChatParams) -> Iterator[Dict]:
        params.load_config(self.model_names[0])
        headers = {
            "Content-Type": "application/json"
        }
        data = {
            "model": params.model_name,
            "messages": params.messages,
            "max_tokens": params.max_tokens,
            "temperature": params.temperature
        }

        url = params.api_url + "/chat/completions"

        response = requests.post(url, headers=headers, json=data)
        ans = response.json()
        content = ans["choices"][0]["message"]["content"]
        yield {"error_code": 0, "text": content}

    def get_embeddings(self, params):
        # 临时解决方案，不支持embedding
        print("embedding")
        print(params)

    def make_conv_template(self, conv_template: str = None, model_path: str = None) -> Conversation:
        return conv.Conversation(
            name=self.model_names[0],
            system_message="你是智谱AI小助手，请根据用户的提示来完成任务",
            messages=[],
            roles=["user", "assistant", "system"],
            sep="\n###",
            stop_str="###",
        )

if __name__ == "__main__":
    import uvicorn
    from server.utils import MakeFastAPIOffline
    from fastchat.serve.model_worker import app

    worker = ChatGLMWorker(
        controller_addr="http://127.0.0.1:20002",
        worker_addr="http://127.0.0.1:21002",
    )
    sys.modules["fastchat.serve.model_worker"].worker = worker
    MakeFastAPIOffline(app)
    uvicorn.run(app, port=21002)