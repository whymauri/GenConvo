"""
Modified from
    https://github.com/HazyResearch/cartridges/blob/main/cartridges/contexts/finance/dataset.py
"""

import requests

from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm


def pdf_to_markdown(pdf_url: str) -> str:
    import pymupdf
    import pymupdf4llm

    # Fetch the PDF content from the URL
    response = requests.get(pdf_url, timeout=30)
    response.raise_for_status()

    # Open the PDF from the fetched content
    pdf_data = response.content
    with pymupdf.open(stream=pdf_data, filetype="pdf") as doc:
        md_text = pymupdf4llm.to_markdown(
            doc, table_strategy="lines", show_progress=False
        )
        return md_text


def _process_url(
    url: str, url_to_name: dict[str, str], output_dir: Path, force: bool
) -> tuple[str, str]:
    path = output_dir / f"{url_to_name[url]}.md"

    if path.exists() and not force:
        text = path.read_text()
    else:
        try:
            text = pdf_to_markdown(url)
            path.write_text(text)
        except Exception as e:
            print(f"Error processing {url}: {e}")
            text = ""
    return url, text


def load_markdown(df: pd.DataFrame, output_dir: Path, force: bool = False):
    """
    Process PDFs for each row in the dataset
    Adds a 'md_text' column with the extracted text
    """
    import multiprocessing as mp
    from functools import partial

    urls = set(df["doc_link"].tolist())
    url_to_name = {row["doc_link"]: row["doc_name"] for _, row in df.iterrows()}

    # Create a partial function with fixed arguments
    process_func = partial(
        _process_url, url_to_name=url_to_name, output_dir=output_dir, force=force
    )

    # Process URLs in parallel
    url_to_text = {}
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = list(
            tqdm(
                pool.imap(process_func, urls),
                total=len(urls),
                desc="Processing PDFs",
                disable=True,
            )
        )

    # Collect results
    url_to_text = dict(results)

    df["md_text"] = df["doc_link"].map(url_to_text, na_action="ignore")

    return df


def load_finance(doc_names: Optional[list[str]] = None, force: bool = False):
    from datasets import load_dataset

    dataset = load_dataset(
        "PatronusAI/financebench", split="train", trust_remote_code=True
    )
    df = dataset.to_pandas()  # type: ignore

    if doc_names is not None:
        df = df[df["doc_name"].isin(doc_names)]

    dataset_dir = Path(dataset.cache_files[0]["filename"]).parent  # type: ignore
    path = dataset_dir / "bench_with_pdfs.feather"
    print(f"Saving datasets {doc_names} to {path}.")

    df = load_markdown(df, dataset_dir, force=force)  # type: ignore
    df.to_feather(path)
    return df
