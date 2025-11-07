from emilia_pipeline import run_emilia_pipeline

results = run_emilia_pipeline(
    "Emilia/config.json",
    input_folder="test_example",
    batch_size=32,
    whisper_arch="large-v3",
    do_uvr=True,
    min_duration=0.25,
    forced_language="en",
)
