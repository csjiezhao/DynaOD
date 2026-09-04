import os
from openai import OpenAI
from dotenv import load_dotenv
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential
)

load_dotenv()

WAIT_TIME_MIN = 1
WAIT_TIME_MAX = 3
ATTEMPT_COUNTER = 3

def get_api_key(platform):
    return os.environ[f'{platform}_API_KEY']


def get_base_url(platform, default=None):
    return os.environ.get(f"{platform}_BASE_URL", default)


class LLMCaller:
    def __init__(self, platform, model_name):
        self.platform = platform
        self.model_dict = {
            'gpt-4o-mini': {
                'OpenAI': 'gpt-4o-mini-2024-07-18',
                'OpenRouter': 'openai/gpt-4o-mini-2024-07-18',
                'DMX': 'gpt-4o-mini-2024-07-18'
            },
            'qwen-2.5-7b':{
                "DMX": "qwen2.5-7b-instruct",
                "DashScope": "qwen2.5-7b-instruct",
                "OpenRouter": "qwen/qwen-2.5-7b-instruct",
            },
            'qwen-2.5-1.5b':{
                "vLLM": "qwen2.5-1.5b-local",
            },
            'qwen-2.5-1.5b-sft': {
                "vLLM": "qwen2.5-1.5b-v1",
            }

        }
        if model_name not in self.model_dict or platform not in self.model_dict[model_name]:
            raise ValueError(f"Unsupported model/platform pair: model={model_name}, platform={platform}")
        self.model_name = self.model_dict[model_name][platform]
        if self.platform == "OpenAI":
            self.client = OpenAI(
                api_key=get_api_key(platform),
                base_url=get_base_url(platform, "https://api.openai.com/v1")
            )
        elif self.platform == "SiliconFlow":
            self.client = OpenAI(
                api_key=get_api_key(platform),
                base_url=get_base_url(platform, "https://api.siliconflow.cn/v1")
            )
        elif self.platform == "DMX":
            self.client = OpenAI(
                api_key=get_api_key(platform),
                base_url=get_base_url(platform, "https://www.dmxapi.cn/v1")
            )
        elif self.platform == "OpenRouter":
            self.client = OpenAI(
                api_key=get_api_key(platform),
                base_url=get_base_url(platform, "https://openrouter.ai/api/v1")
            )
        elif self.platform == "DashScope":
            self.client = OpenAI(
                api_key=get_api_key(platform),
                base_url=get_base_url(platform, "https://dashscope.aliyuncs.com/compatible-mode/v1")
            )
        elif self.platform == 'vLLM':
            self.client = OpenAI(
                base_url=get_base_url(platform, "http://127.0.0.1:8000/v1"),
                api_key=get_api_key(platform),
            )


    @retry(wait=wait_random_exponential(min=WAIT_TIME_MIN, max=WAIT_TIME_MAX), stop=stop_after_attempt(ATTEMPT_COUNTER))
    def get_response(self, messages, max_tokens=1024, temperature=0., return_json=False):
        params = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        if return_json:
            params["response_format"] = {"type": "json_object"}

        completion = self.client.chat.completions.create(**params)
        return completion.choices[0].message.content.strip()


if __name__ == '__main__':
    llm = LLMCaller('vLLM', 'qwen-2.5-1.5b-sft')
    m1 = """
    You are an expert in urban mobility and spatial economics.
    Your task is to generate a POI control vector for a specific USA region (identified by its Tract GEOID) on a given day, considering how time-dependent factors (like day of the week, holidays, etc.) affect POI usage.

    Use your knowledge to reason how human activity and POI usage would vary across different days (weekdays vs weekends, holidays, seasonal trends).
    Do not produce "all-zero" outputs unless absolutely justified.

    ### Input:
    - **Tract GEOID**:
    18077966000
    This GEOID represents the location of the tract, helping understand its geographical context.

    - **POI features** for the region:
    "finance", "public", "transport", "entertainment", "health", "service", "education", "government",
    "religion", "accommodation", "food", "cafe", "fast_food", "ice_cream", "pub", "restaurant",
    "shop_beauty", "shop_clothes", "boutique", "shop_transport", "retail", "commodity", "marketplace",
    "home-improvement", "sport", "public_transport", "kindergarten", "office", "recycling",
    "travel_agency", "tourism", "shop_livelihood", "residential", "dormitory"

    - **Date**:
    2019_01_22
    The date affects POI usage. Consider how behavior changes over time (e.g., weekdays vs weekends, workdays vs holidays).

    ### Output:
    - Output a **single JSON object**.
    - **Key** = Tract GEOID.
    - **Value** = A list of 34 integers corresponding to POI categories.
      - Allowed values: 1 (increase), 0 (no_change), -1 (decrease).
    - Do not include explanations, comments, or extra text.

    ### Output format:
    {{
    "18077966000": [34 integers here]
    }}
    """

    m2 = """You are given a US tract GEOID and a date.
    TASK: POI_CONTROL
    Output JSON only. No extra text.
    Key MUST be TRACT. Value MUST be a list of EXACTLY 34 integers in (-1,0,1).

    TRACT: 18077966000
    DATE: 2019_01_22
    """

    messages = [{"role": "user", "content": m1}]
    print(llm.get_response(messages))
