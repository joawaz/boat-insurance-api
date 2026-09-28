import os
import json
import base64

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from openai import OpenAI


app = FastAPI()


# ============================================================
# CONFIGURATION
# ============================================================

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
API_SECRET = os.environ["API_SECRET"]

# Keep the model configurable.
# Current default: GPT-6 Luna
OPENAI_MODEL = os.environ.get(
    "OPENAI_MODEL",
    "gpt-6-luna"
)

client = OpenAI(api_key=OPENAI_API_KEY)


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
# PDF EXTRACTION
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
    # Basic size protection
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
    # Convert PDF to base64
    # --------------------------------------------------------

    pdf_base64 = base64.b64encode(
        pdf_bytes
    ).decode("ascii")


    # --------------------------------------------------------
    # Extraction instructions
    # --------------------------------------------------------

    instructions = """
You extract structured information from boat insurance
documents.

The document may come from any insurance company.

The layout can be completely different between companies.

The document may be:
- Swedish
- English
- mixed Swedish and English
- digitally generated
- scanned

Look through the entire PDF.

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
   numbers as they appear in the document.

10. Do not guess missing characters in identifiers.

11. If several boats are mentioned, identify the boat that
    is actually insured by this policy. If this cannot be
    determined reliably, return null for the affected fields.

12. The insurance company is the company providing the
    insurance, not a broker, bank, boat dealer or payment
    provider.

Return only the requested JSON structure.
"""


    # --------------------------------------------------------
    # JSON schema
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Call OpenAI
    # --------------------------------------------------------

    try:

        response = client.responses.create(

            model=OPENAI_MODEL,

            input=[
                {
                    "role": "user",

                    "content": [

                        {
                            "type": "input_file",

                            "filename": filename,

                            "file_data":
                                "data:application/pdf;base64,"
                                + pdf_base64
                        },

                        {
                            "type": "input_text",

                            "text": instructions
                        }
                    ]
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


        # ----------------------------------------------------
        # Parse JSON
        # ----------------------------------------------------

        result = json.loads(
            response.output_text
        )


        # ----------------------------------------------------
        # Return result
        # ----------------------------------------------------

        return JSONResponse(
            content={
                "success": True,
                "data": result
            }
        )


    except Exception as e:

        return JSONResponse(

            status_code=500,

            content={
                "success": False,
                "error": str(e)
            }
        )