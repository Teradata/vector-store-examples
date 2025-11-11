import re
import zipfile
import logging
from typing import Iterable, Optional, Sequence
import os

from huggingface_hub import hf_hub_download, list_repo_files

import teradataml as tdml

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _sanitize_repo_id(repo_id: str) -> str:
    """
    Convert 'org/name' -> 'models--org--name' for a filesystem-friendly directory.
    """
    return "models--" + repo_id.replace("/", "--")


def download_hf_model_to_dir(
    repo_id: str,
    local_root: str,
    *,
    filenames: Optional[Sequence[str]] = None,
    include_patterns: Optional[Iterable[str]] = None,
    exclude_patterns: Optional[Iterable[str]] = None,
    revision: Optional[str] = None,
    token: Optional[str] = None,
    repo_type: str = "model",
) -> str:
    """
    Download files from a Hugging Face repo into a deterministic local directory using hf_hub_download.

    - If `filenames` is provided, only those files are downloaded.
    - Otherwise, we list the repo and pull files matching `include_patterns` (and not matching `exclude_patterns`).
    - Files are *copied* (no symlinks) into `<local_root>/<models--org--name>`.

    Returns the absolute path to the local model directory.
    """
    valid_name = _sanitize_repo_id(repo_id)
    model_dir = os.path.abspath(os.path.join(local_root, valid_name))
    os.makedirs(model_dir, exist_ok=True)
    logger.info(f"Target model directory: {model_dir}")

    # Figure out which files to download
    if filenames is None:
        logger.info("No explicit filenames provided; listing repo to select files...")
        all_files = list_repo_files(repo_id=repo_id, revision=revision, repo_type=repo_type, token=token)

        def _matches_any(path: str, patterns: Iterable[str]) -> bool:
            return any(re.search(p, path) for p in patterns)

        candidates = all_files
        if include_patterns:
            candidates = [f for f in candidates if _matches_any(f, include_patterns)]
        if exclude_patterns:
            candidates = [f for f in candidates if not _matches_any(f, exclude_patterns)]

        filenames = sorted(candidates)
        logger.info(f"{len(filenames)} file(s) selected for download.")
        if not filenames:
            raise ValueError("No files matched the provided include/exclude patterns.")
    else:
        filenames = list(filenames)

    # Download files with real copies into model_dir (no symlinks)
    for fn in filenames:
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=fn,
            revision=revision,
            token=token,
            repo_type=repo_type,
            local_dir=model_dir,
            local_dir_use_symlinks=False,  # ensure real files are copied into target dir
        )
        logger.info(f"Downloaded: {fn} -> {local_path}")

    return model_dir


def zip_saved_files(repo_id: str, local_dir: str) -> str:
    """
    Zip the *contents* of `local_dir` into ./models/<models--org--name>.zip
    (keeps relative paths inside the archive).
    """
    valid_name = _sanitize_repo_id(repo_id)
    models_dir = os.path.join(".", "models")
    os.makedirs(models_dir, exist_ok=True)

    zip_path = os.path.abspath(os.path.join(models_dir, f"{valid_name}.zip"))
    logger.info(f"Creating zip archive at: {zip_path}")

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(local_dir):
            for file in files:
                fp = os.path.join(root, file)
                arcname = os.path.relpath(fp, start=local_dir)
                zf.write(fp, arcname)

    logger.info(f"Zip created: {zip_path}")
    return zip_path

def fetch_and_zip_hf_model(
    repo_id: str,
    local_root: str = "./models",
    include_patterns=None,
    exclude_patterns=None,
):
    """
    Download a Hugging Face model into a directory and zip the saved files.

    Returns:
        tuple[str, str, str]:
            (output_directory, zip_path, zip_filename)
    """
    if include_patterns is None:
        include_patterns = [
            r"(config\.json|pytorch_model\.bin|model\.safetensors|"
            r"tokenizer(\.json|\.config\.json)?|vocab\.txt)$"
        ]

    if exclude_patterns is None:
        exclude_patterns = [r"\.lock$"]

    logger.info(f"Starting model fetch for '{repo_id}'")
    logger.debug(f"Include patterns: {include_patterns}")
    logger.debug(f"Exclude patterns: {exclude_patterns}")

    # Download filtered model files
    logger.info("Downloading model files...")
    out_dir = download_hf_model_to_dir(
        repo_id=repo_id,
        local_root=local_root,
        include_patterns=include_patterns,
        exclude_patterns=exclude_patterns,
    )
    logger.info(f"Files saved to directory: {out_dir}")

    # Make a safe directory name (e.g., "openai--whisper-tiny")
    safe_name = repo_id.replace("/", "--")

    # Zip the model files
    logger.info("Zipping downloaded model files...")
    zip_path = zip_saved_files(safe_name, out_dir)

    # Extract just the file name
    zip_filename = os.path.basename(zip_path)

    logger.info(f"Zip created at: {zip_path}")
    logger.info(f"Zip filename: {zip_filename}")
    logger.info("Model fetch + zip complete.")

    return out_dir, zip_path, zip_filename

def install_model_zip(model_zip: str):
    """
    Install a model zip using TDML after removing any existing version.
    model_id is automatically derived from the zip name:
        e.g.  "whisper-tiny.zip" -> "whisper_tiny"
    """

    zip_path = os.path.join("./models", model_zip)

    # derive ID: remove extension, replace "-" with "_"
    model_root = os.path.splitext(model_zip)[0]
    model_id = model_root.replace("-", "_")

    logger.info(f"Preparing to install model zip '{model_zip}' (model_id='{model_id}')")

    if not os.path.exists(zip_path):
        logger.error(f"File not found: {zip_path}")
        return False

    # Remove previous model version
    logger.info(f"Removing existing model with id='{model_id}' (if exists)")
    try:
        tdml.remove_file(file_identifier=model_id, force_remove=True)
        logger.info("Existing model removed.")
    except Exception as e:
        logger.warning(f"Could not remove existing model: {str(e).splitlines()[0]}")

    # Install new model zip
    logger.info(f"Installing model from: {zip_path}")
    try:
        tdml.install_file(
            file_identifier=model_id,
            file_path=zip_path,
            is_binary=True,
        )
        logger.info("Model zip installed successfully.")
        return True
    except Exception as e:
        logger.error(f"Installation failed: {e}")
        return False


import base64
import logging
from pathlib import Path
import wave
import contextlib

import pandas as pd
from tqdm.auto import tqdm


def wavs_to_dataframe(
    directory,
    include_base64=True,
    include_bytes=False,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """
    Serialize .wav files in `directory` (non-recursive) into a pandas DataFrame.

    Parameters
    ----------
    directory : str | Path
        Folder to scan for .wav files (non-recursive).
    include_base64 : bool, default True
        Include base64-encoded file payload in 'data_b64'.
    include_bytes : bool, default False
        Include raw file bytes in 'data_bytes' (can be large).
    logger : logging.Logger | None
        If None, a default logger is created.

    Returns
    -------
    pd.DataFrame with columns:
      path, filename, samplerate, channels, sampwidth_bytes, n_frames,
      duration_sec, data_b64 (optional), data_bytes (optional)
    """
    log = logger or logging.getLogger("wavs_to_dataframe")
    if not logger:  # minimal, quiet default
        if not log.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
            log.addHandler(handler)
        log.setLevel(logging.INFO)

    directory = Path(directory).expanduser().resolve()
    files = sorted(directory.glob("*.wav"))
    log.info("Found %d .wav files in %s", len(files), directory)

    rows = []
    for p in tqdm(files, desc="Processing WAVs"):
        try:
            with contextlib.closing(wave.open(str(p), "rb")) as wf:
                n_channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                framerate = wf.getframerate()
                n_frames = wf.getnframes()
                duration = (n_frames / float(framerate)) if framerate else 0.0

            data_b64 = None
            data_bytes = None
            if include_base64 or include_bytes:
                raw = p.read_bytes()
                if include_bytes:
                    data_bytes = raw
                if include_base64:
                    data_b64 = base64.b64encode(raw).decode("ascii")

            rows.append(
                {
                    "path": str(p),
                    "filename": p.name,
                    "samplerate": framerate,
                    "channels": n_channels,
                    "sampwidth_bytes": sampwidth,
                    "n_frames": n_frames,
                    "duration_sec": duration,
                    "data_b64": data_b64,
                }
            )
        except Exception as e:
            log.warning("Skipping %s due to error: %s", p, e)

    df = pd.DataFrame(rows)
    log.info("Created DataFrame with %d rows", len(df))
    return df


def install_sto_script(filename: str, database: str) -> bool:
    """
    Remove the existing filemane script and install a new one.
    It uses the filename (without extension) as the file identifier.
    (and replaces '-' with '_' in the identifier for sake of compatibility)

    Parameters
    ----------
    filename : str
        The local path to the script to install (e.g. 'STO_memory.py').
    database : str
        The target database where the script will be installed.

    Returns
    -------
    bool
        True if installation succeeded, False otherwise.
    """

    tdml.execute_sql(f'SET SESSION SEARCHUIFDBPATH = "{database}"')
    logger.info(f'SET SESSION SEARCHUIFDBPATH = "{database}"')
    tdml.execute_sql(f'DATABASE {database}')
    logger.info(f'DATABASE {database}')

                            
    # Remove old file
    file_identifier='.'.join(filename.split('/')[-1].split('.')[:-1]).replace('-','_')
    logger.info(f'file_identifier: {file_identifier}')
    try:
        tdml.remove_file(file_identifier=file_identifier, force_remove=True)
        logger.info("Removed existing STO script")
    except Exception as e:
        error_msg = str(e).split('\n')[0]
        logger.warning(f"Could not remove existing STO script: {error_msg}")

    # Install new script
    if os.path.exists(filename):
        try:
            tdml.install_file(
                file_identifier = file_identifier,
                file_path       = filename,
                is_binary       = False
            )
            logger.info(f"STO script {filename} installed successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to install STO script {filename}: {e}")
            return False
    else:
        logger.error(f"{filename} not found")
        return False
    
def execute_sto_script(model_zip: str, database: str, script: str = 'whisper_transcribe.py', data_table: str = 'T_WAV_FILES', target_table: str = 'T_TRANSCRIPT_TABLE', if_exists: str = 'replace') -> list:
    """
    Execute the STO script 'whisper_transcribe.py' installed in the system database.

    Parameters
    ----------
    model_zip : str
        The name of the model zip file to be used in the script.
    database : str, optional
        The database where the script and data table are located (default is settings.system_db).
    script : str, optional
        The name of the STO script to execute (default is 'whisper_transcribe.py').
    data_table : str, optional
        The name of the data table containing input data (default is 'T_WAV_FILES')
    target_table : str, optional
        The name of the target table to store results (default is 'T_TRANSCRIPT_TABLE').
    if_exists : str, optional
        Action to take if the target table exists ('replace' or 'append', default is 'replace').

    Returns
    -------
    None
    """
    tdml.execute_sql(f'SET SESSION SEARCHUIFDBPATH = "{database}"')
    logger.info(f'SET SESSION SEARCHUIFDBPATH = "{database}"')
    tdml.execute_sql(f'DATABASE {database}')
    logger.info(f'DATABASE {database}')

    column_names     = [
        'path',
        'filename',
        'samplerate',
        'channels',
        'sampwidth_bytes',
        'n_frames',
        'duration_sec',
        'data_b64'
        ]
    
    # Test column names of data_table
    df = tdml.DataFrame(tdml.in_schema(database, data_table))
    if list(df.columns) != column_names:
        logger.error(f"Column names of {data_table} do not match expected names.")
        logger.error(f"Expected: {column_names}")
        logger.error(f"Found: {list(df.columns)}")
        raise ValueError("Column names do not match expected names.")
    

    query = f"""
                SELECT *
                FROM SCRIPT (
                    ON (SELECT * FROM {database}.{data_table}) AS INPUT_TABLE
                    HASH BY filename
                    SCRIPT_COMMAND('mkdir $PWD/models && unzip {database}/{model_zip} -d $PWD/models/{model_zip.split('.')[0]}/ > /dev/null && tdpython3 {database}/{script} "models/{model_zip.split('.')[0]}"')
                    RETURNS (
                    'filename VARCHAR(255)',
                    'status VARCHAR(50)',
                    'transcription CLOB(2097088000)'
                    )
                ) AS d
    """
    
    query_ = f"""
                SELECT DISTINCT filename
                FROM SCRIPT (
                    ON (SELECT * FROM {database}.{data_table}) AS INPUT_TABLE
                    HASH BY filename
                    SCRIPT_COMMAND('tdpip3 freeze ')
                    RETURNS ('filename VARCHAR(500)')
                ) AS d
                ORDER BY 1
                WHERE filename like 'transformers%' or filename like 'tokenizers%' or filename like 'torch%' or filename like 'sentencepiece%'
    """

    query__   = f"""
                SELECT DISTINCT *
                FROM SCRIPT (
                    ON (SELECT * FROM {database}.{data_table}) AS INPUT_TABLE
                    HASH BY filename
                    SCRIPT_COMMAND('tdpython3 --version')
                    RETURNS ('filename VARCHAR(500)')
                ) AS d
                ORDER BY 1
    """

    #logger.info(f'STO query generated:\n{query}')
    if if_exists.lower() == 'replace':
        try:
            tdml.execute_sql(f'DROP TABLE {database}.{target_table}')
        except:
            pass
        
        qry_sto = f''' 
            CREATE MULTISET TABLE {database}.{target_table} AS (
            {query}
            ) WITH DATA;
            '''

        tdml.execute_sql(qry_sto)
        logger.info("STO execution completed successfully")
    elif if_exists == 'append':
        qry_sto = f''' 
            INSERT INTO {database}.{target_table} 
            {query}
            '''

        tdml.execute_sql(qry_sto)
        logger.info("STO execution completed successfully")
    return tdml.DataFrame(tdml.in_schema(database, target_table))   
