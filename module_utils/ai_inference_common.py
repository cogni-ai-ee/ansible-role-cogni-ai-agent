from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import AssistantMessage, SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential


def create_client(endpoint, token):
    return ChatCompletionsClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(token),
    )


def build_messages(raw_messages=None, system_prompt=None, prompt=None):
    messages = []

    if system_prompt:
        messages.append(SystemMessage(system_prompt))

    if prompt:
        messages.append(UserMessage(prompt))

    for item in raw_messages or []:
        role = item.get("role")
        content = item.get("content")

        if role == "system":
            messages.append(SystemMessage(content))
        elif role == "user":
            messages.append(UserMessage(content))
        elif role == "assistant":
            messages.append(AssistantMessage(content))
        else:
            raise ValueError("Unsupported role: %s" % role)

    return messages
