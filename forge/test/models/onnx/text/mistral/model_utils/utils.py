# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


def get_current_weather(location: str, format: str):
    """
    Get the current weather

    Args:
        location: The city and state, e.g. San Francisco, CA
        format: The temperature unit to use. Infer this from the users location. (choices: ["celsius", "fahrenheit"])
    """
