from google import genai  # make sure this is installed and authenticated properly


def generate_stem_analysis(
    file_path: str,
    prompt: str,
    model_name: str = "gemini-2.0-flash-exp-image-generation",
    api_key: str = None,
):
    """
    Uploads a STEM image and queries the Gemini model with a given prompt.

    Args:
        file_path (str): Path to the image file.
        prompt (str): Instruction to Gemini model.
        model_name (str): Gemini model name (default: "gemini-2.0-flash-exp-image-generation").
        api_key (str): API key for Gemini (if not already set via environment).

    Returns:
        str: Response text from the Gemini model.
    """
    if api_key is None:
        raise ValueError("API key must be provided")

    client = genai.Client(api_key=api_key)

    try:
        uploaded_file = client.files.upload(file=file_path)
    except Exception as e:
        raise RuntimeError(f"File upload failed: {e}")

    try:
        response = client.models.generate_content(
            model=model_name, contents=[uploaded_file, prompt]
        )
        return response.text
    except Exception as e:
        raise RuntimeError(f"Content generation failed: {e}")
