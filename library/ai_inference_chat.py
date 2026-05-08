#!/usr/bin/python

from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.ai_inference_common import create_client, build_messages
import os


def main():
    module = AnsibleModule(
        argument_spec=dict(
            endpoint=dict(type="str", default="https://models.github.ai/inference"),
            token=dict(type="str", no_log=True, required=False),
            model=dict(type="str", default="deepseek/DeepSeek-R1"),
            prompt=dict(type="str", required=False),
            system_prompt=dict(type="str", required=False),
            messages=dict(type="list", elements="dict", required=False, default=[]),
            max_tokens=dict(type="int", required=False),
            temperature=dict(type="float", required=False),
        ),
        required_one_of=[["prompt", "messages"]],
        supports_check_mode=True,
    )

    if module.check_mode:
        module.exit_json(changed=False, skipped=True, msg="Check mode: no request sent")

    token = module.params["token"] or os.getenv("GITHUB_TOKEN")
    if not token:
        module.fail_json(msg="token is required or set GITHUB_TOKEN")

    try:
        client = create_client(module.params["endpoint"], token)
        messages = build_messages(
            raw_messages=module.params["messages"],
            system_prompt=module.params["system_prompt"],
            prompt=module.params["prompt"],
        )

        kwargs = {
            "messages": messages,
            "model": module.params["model"],
        }

        if module.params["max_tokens"] is not None:
            kwargs["max_tokens"] = module.params["max_tokens"]

        if module.params["temperature"] is not None:
            kwargs["temperature"] = module.params["temperature"]

        response = client.complete(**kwargs)

        module.exit_json(
            changed=False,
            message=response.choices[0].message.content,
            model=module.params["model"],
            messages_count=len(messages),
        )
    except Exception as exc:
        module.fail_json(msg=str(exc))


if __name__ == "__main__":
    main()
