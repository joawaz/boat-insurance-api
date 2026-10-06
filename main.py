import os
import json
import base64
import asyncio
import re

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import PlainTextResponse
from openai import OpenAI, RateLimitError

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None


app = FastAPI()


# ============================================================
# CONFIGURATION
# ============================================================

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
API_SECRET = os.environ["API_SECRET"]

OPENAI_MODEL = os.environ.get(
    "OPENAI_MODEL",
    "gpt-6-luna"
)

# Keep requests serialized on a single Render instance.
# This prevents several PDFs from consuming the TPM allowance
# simultaneously.
OPENAI_CONCURRENCY = int(os.environ.get("OPENAI_CONCURRENCY", "1"))

# Maximum amount of relevant extracted PDF text sent to the model.
# Large PDFs are reduced by insurance-specific keyword selection.
MAX_TEXT_CHARS = int(os.environ.get("MAX_TEXT_CHARS", "60000"))

# If a PDF contains little/no extractable text, fall back to sending
# the PDF itself so scanned/image PDFs can still be processed.
MIN_EXTRACTED_TEXT_CHARS = int(
    os.environ.get("MIN_EXTRACTED_TEXT_CHARS", "500")
)

# Do not spend hours retrying a rate limit. The caller can retry later.
MAX_RETRY_SECONDS = int(os.environ.get("MAX_RETRY_SECONDS", "120"))

client = OpenAI(api_key=OPENAI_API_KEY)

openai_semaphore = asyncio.Semaphore(OPENAI_CONCURRENCY)


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/")
def root():
    return {
        "service": "boat-insurance-api",
        "status": "ok"
    }


# ============================================================
# PDF TEXT EXTRACTION
# ============================================================

def extract_pdf_text(pdf_bytes):
    """
    Extract text locally from the PDF and keep page boundaries.

    Returns:
        (text, page_count)

    Page markers make it possible to select relevant sections later
    without simply taking an arbitrary beginning/end slice.
    """
    if PdfReader is None:
        return "", 0

    try:
        import io

        reader = PdfReader(io.BytesIO(pdf_bytes))
        page_count = len(reader.pages)
        pages = []

        for number, page in enumerate(reader.pages, start=1):
            try:
                page_text = page.extract_text() or ""
            except Exception:
                page_text = ""

            page_text = re.sub(r"[ \t]+", " ", page_text)
            page_text = re.sub(r"\n{3,}", "\n\n", page_text).strip()

            if page_text:
                pages.append(
                    f"\n--- PAGE {number} ---\n{page_text}"
                )

        return "\n".join(pages).strip(), page_count

    except Exception:
        return "", 0


def build_relevant_text(text, max_chars):
    """
    Reduce a large extracted document using insurance-specific keywords.

    This deliberately does NOT use a simple first-N/last-N truncation.
    It keeps:
      * the beginning/end of the document
      * pages containing insurance/policy/boat/identifier/period terms
      * nearby text around those terms

    If the document is already small enough, all text is returned.
    """
    if len(text) <= max_chars:
        return text

    lines = text.splitlines()

    keyword_pattern = re.compile(
        r""
        r"försäkringstagare|insured|försäkrad|"
        r"company|bolag|försäkringsbolag|försäkringsgivare|"
        r"insurance|försäkring|försäkrings|policy|försäkringsnummer|båtförsäkring|"
        r"coverage|gäller|period|perioden|from|to|från|till|tiden|försäkringstid|försäkringsperiod|"
        r"tillverkare|brand|märke|hersteller|fabrikat|"
        r"boat|båt|vessel|fartyg|make|model|modell|"
        r"registration|registrering|registreringsnummer|hull|skrov|hin|cin|s/n|serienummer|tillverkningsnr|skrovnummer|"
        r"year|årsmodell|byggår|år|tillverkningsår|"
        r"",
        re.IGNORECASE | re.VERBOSE
    )



    selected = set()

    # Always retain useful document boundaries.
    for i in range(min(80, len(lines))):
        selected.add(i)
    for i in range(max(0, len(lines) - 50), len(lines)):
        selected.add(i)

    # Keep context around every relevant line.  A window of 4 lines
    # usually captures the label/value pair without pulling in pages
    # of unrelated terms and conditions.
    for i, line in enumerate(lines):
        if keyword_pattern.search(line):
            for j in range(max(0, i - 4), min(len(lines), i + 5)):
                selected.add(j)

    # Preserve page markers associated with selected content.
    candidate = []
    for i in sorted(selected):
        line = lines[i].strip()
        if line:
            candidate.append(line)

    relevant = "\n".join(candidate)

    if len(relevant) <= max_chars:
        return relevant

    # If keyword selection is still large, rank lines by relevance and
    # retain the highest-value lines while preserving document order.
    scored = []
    for i, line in enumerate(lines):
        if not line.strip():
            continue

        score = 0
        matches = keyword_pattern.findall(line)
        score += len(matches) * 5

        # Stronger signals for fields we actually need.
        if re.search(r"policy|försäkringsnummer|policynummer", line, re.I):
            score += 8
        if re.search(r"gäller|coverage|period|från|till|from|to", line, re.I):
            score += 8
        if re.search(r"hull|skrov|hin|cin|registration|registrering", line, re.I):
            score += 7
        if re.search(r"boat|båt|vessel|make|märke|model|modell", line, re.I):
            score += 6
        if re.search(r"year|årsmodell|byggår", line, re.I):
            score += 5

        if score:
            scored.append((score, i, line))

    # Highest scoring lines first, then restore document order.
    scored.sort(key=lambda x: (-x[0], x[1]))

    chosen = set()
    used = 0
    for score, i, line in scored:
        cost = len(line) + 1
        if used + cost > max_chars:
            continue
        chosen.add(i)
        used += cost

    final_lines = [lines[i].strip() for i in sorted(chosen) if lines[i].strip()]
    result = "\n".join(final_lines)

    return result[:max_chars]


# ============================================================
# EXTRACTION INSTRUCTIONS
# ============================================================

instructions = """
You extract information from boat insurance documents.

The document can come from ANY insurance company.

Different insurance companies can have completely different
document layouts.

The document may be:
- Swedish
- English
- mixed Swedish and English
- digitally generated
- scanned

Look through the available document information carefully.

IMPORTANT RULES:

1. Never invent information.

2. If a value cannot be established reliably from the
   document, return null.

3. Extract the insurance company.

4. Extract the insurance policy number.

5. Extract the insured boat:
   - make
   - model
   - registration number
   - hull number / HIN / CIN
   - year

6. Extract the ACTUAL INSURANCE COVERAGE PERIOD.

7. Do NOT confuse the insurance period with:
   - document date
   - issue date
   - payment date
   - invoice date
   - renewal date
   - claim date
   - transaction date

8. Dates must be returned as YYYY-MM-DD.

9. Preserve registration numbers, hull numbers and policy
   numbers exactly as they appear.

10. Do not guess missing characters in identifiers.

11. If several boats are mentioned, identify the boat that
    is actually insured by this policy.

12. The insurance company is the company providing the
    insurance, not a broker, bank, boat dealer or payment
    provider.

13. If the supplied text is truncated, only use information
    that is actually present. Do not guess information that
    may have been in the omitted part.

Return ONLY the requested structured data.
"""


# ============================================================
# STRUCTURED JSON SCHEMA
# ============================================================

schema = {
    "type": "object",
    "additionalProperties": False,

    "properties": {

        "insurance_company": {
            "type": ["string", "null"]
        },

        "policy_number": {
            "type": ["string", "null"]
        },

        "boat": {
            "type": "object",
            "additionalProperties": False,

            "properties": {

                "make": {
                    "type": ["string", "null"]
                },

                "model": {
                    "type": ["string", "null"]
                },

                "registration_number": {
                    "type": ["string", "null"]
                },

                "hull_number": {
                    "type": ["string", "null"]
                },

                "year": {
                    "type": ["integer", "null"]
                }
            },

            "required": [
                "make",
                "model",
                "registration_number",
                "hull_number",
                "year"
            ]
        },

        "insurance_period": {
            "type": "object",
            "additionalProperties": False,

            "properties": {

                "start": {
                    "type": ["string", "null"]
                },

                "end": {
                    "type": ["string", "null"]
                }
            },

            "required": [
                "start",
                "end"
            ]
        }
    },

    "required": [
        "insurance_company",
        "policy_number",
        "boat",
        "insurance_period"
    ]
}


# ============================================================
# OPENAI CALL
# ============================================================

def create_openai_response(pdf_bytes, filename, extracted_text):
    """
    Use text extraction for normal PDFs because it is much cheaper
    in input tokens than sending the complete PDF.

    For scanned/image-only PDFs, fall back to PDF input.
    """

    if len(extracted_text.strip()) >= MIN_EXTRACTED_TEXT_CHARS:
        text_for_model = build_relevant_text(
            extracted_text,
            MAX_TEXT_CHARS
        )

        input_content = [
            {
                "type": "input_text",
                "text": instructions
            },
            {
                "type": "input_text",
                "text": (
                    "Here is the extracted text from the insurance "
                    "document:\n\n" + text_for_model
                )
            }
        ]

    else:
        pdf_base64 = base64.b64encode(
            pdf_bytes
        ).decode("ascii")

        input_content = [
            {
                "type": "input_text",
                "text": instructions
            },
            {
                "type": "input_file",
                "filename": filename,
                "file_data": (
                    "data:application/pdf;base64,"
                    + pdf_base64
                )
            }
        ]

    return client.responses.create(
        model=OPENAI_MODEL,

        input=[
            {
                "role": "user",
                "content": input_content
            }
        ],

        text={
            "format": {
                "type": "json_schema",
                "name": "boat_insurance",
                "strict": True,
                "schema": schema
            }
        }
    )


async def call_openai_with_limit(pdf_bytes, filename, extracted_text):
    """
    Serialize calls and handle 429 without turning it into HTTP 500.
    """

    async with openai_semaphore:

        try:
            return create_openai_response(
                pdf_bytes,
                filename,
                extracted_text
            )

        except RateLimitError as e:
            message = str(e)

            # The OpenAI error may contain a retry duration.
            # We intentionally do not sleep for hours.
            retry_seconds = 60

            match = re.search(
                r"try again in ([0-9]+)s",
                message,
                re.IGNORECASE
            )

            if match:
                retry_seconds = int(match.group(1))

            retry_seconds = min(
                retry_seconds,
                MAX_RETRY_SECONDS
            )

            raise HTTPException(
                status_code=429,
                detail={
                    "error": "OpenAI rate limit reached",
                    "retry_after_seconds": retry_seconds,
                    "message": message
                }
            )

        except Exception:
            raise


# ============================================================
# BOAT INSURANCE EXTRACTION
# ============================================================

@app.post("/boat-insurance")
async def boat_insurance(request: Request):

    # --------------------------------------------------------
    # Check API key
    # --------------------------------------------------------

    supplied_key = request.headers.get("X-API-Key")

    if not supplied_key or supplied_key != API_SECRET:
        raise HTTPException(
            status_code=401,
            detail="Unauthorized"
        )


    # --------------------------------------------------------
    # Read PDF
    # --------------------------------------------------------

    pdf_bytes = await request.body()

    if not pdf_bytes:
        raise HTTPException(
            status_code=400,
            detail="Empty request body"
        )


    # --------------------------------------------------------
    # Limit PDF size
    # --------------------------------------------------------

    max_size = 20 * 1024 * 1024

    if len(pdf_bytes) > max_size:
        raise HTTPException(
            status_code=413,
            detail="PDF is larger than 20 MB"
        )


    # --------------------------------------------------------
    # Check content type
    # --------------------------------------------------------

    content_type = request.headers.get(
        "Content-Type",
        ""
    ).lower()

    if "application/pdf" not in content_type:
        raise HTTPException(
            status_code=400,
            detail="Content-Type must be application/pdf"
        )


    filename = request.headers.get(
        "X-Filename",
        "insurance.pdf"
    )


    # --------------------------------------------------------
    # Extract PDF text locally
    # --------------------------------------------------------

    extracted_text, page_count = extract_pdf_text(
        pdf_bytes
    )


    # --------------------------------------------------------
    # Call OpenAI
    # --------------------------------------------------------

    try:

        response = await call_openai_with_limit(
            pdf_bytes,
            filename,
            extracted_text
        )


        # ----------------------------------------------------
        # Parse OpenAI JSON
        # ----------------------------------------------------

        result = json.loads(
            response.output_text
        )


        # ----------------------------------------------------
        # Extract fields
        # ----------------------------------------------------

        insurance_company = result.get(
            "insurance_company"
        )

        policy_number = result.get(
            "policy_number"
        )

        boat = result.get(
            "boat",
            {}
        )

        insurance_period = result.get(
            "insurance_period",
            {}
        )


        boat_make = boat.get("make")

        boat_model = boat.get("model")

        registration_number = boat.get(
            "registration_number"
        )

        hull_number = boat.get(
            "hull_number"
        )

        boat_year = boat.get("year")

        insurance_start = insurance_period.get(
            "start"
        )

        insurance_end = insurance_period.get(
            "end"
        )


        # ----------------------------------------------------
        # Convert None to empty string
        # ----------------------------------------------------

        values = [

            insurance_company,
            policy_number,
            boat_make,
            boat_model,
            registration_number,
            hull_number,
            boat_year,
            insurance_start,
            insurance_end
        ]


        values = [
            "" if value is None else str(value)
            for value in values
        ]


        # ----------------------------------------------------
        # Protect pipe-delimited format
        # ----------------------------------------------------

        values = [
            value.replace("|", "/")
            for value in values
        ]


        # ----------------------------------------------------
        # Return one single line
        # ----------------------------------------------------

        csv_line = "|".join(values)

        return PlainTextResponse(
            content=csv_line,
            media_type="text/plain"
        )


    except HTTPException:
        raise


    except Exception as e:

        return PlainTextResponse(
            content=(
                "ERROR|"
                + str(e).replace("|", "/")
            ),
            status_code=500,
            media_type="text/plain"
        )
