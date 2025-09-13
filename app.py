from flask import Flask, request, jsonify, render_template, send_file, session, redirect, url_for
from functools import wraps
import os
from datetime import datetime

# ---------------- Additional Imports for OCR/Parsing Layer -----------------
import io
import re
import csv
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from werkzeug.utils import secure_filename

try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except Exception:
    # Fallback if Tesseract/PIL are not available in the environment
    OCR_AVAILABLE = False
# Optional: EasyOCR for better handwriting support
try:
    import easyocr  # type: ignore
    EASY_OCR_AVAILABLE = True
except Exception:
    EASY_OCR_AVAILABLE = False
# --------------------------------------------------------------------------

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-change-me")

# A simple dictionary to simulate AI suggestions based on keywords
suggestion_map = {
    "spam": ["unwanted", "repetitive", "spam"],
    "nudity": ["nude", "sexual", "explicit", "naked", "exposing"],
    "hate": ["hate", "vilify", "incite", "discrimination", "loser", "go die", "unwanted"],
    "violence": ["violence", "dangerous", "attack", "punch", "fight"],
    "illegal": ["illegal", "regulated", "drugs", "offence", "sale"],
    "bullying": ["bully", "harass", "threat", "loser"],
    "ip-violation": ["copyright", "trademark", "intellectual"],
    "suicide": ["suicide", "self-harm", "injury", "kill yourself"],
    "eating-disorders": ["anorexia", "bulimia", "eating"],
    "deepfakes": ["manipulated", "celebrity"]
}

# ------------------- PDO OCR + Parsing Layer (Prototype) -------------------

# Normalisation for informal answers -> system-compliant values
NORMALISE_MAP = {
    "not sure": "Unknown",
    "dont know": "Unknown",
    "don't know": "Unknown",
    "dk": "Unknown",
    "idk": "Unknown",
    "maybe": "Unknown",
    "n/a": "Not Applicable",
    "na": "Not Applicable",
    "nil": "Not Provided",
    "-": "Not Provided",
    "none": "Not Provided",
    "sg": "Singaporean",
}

# Very lightweight schema for common PDO form fields (extend as needed)
@dataclass
class FieldSpec:
    name: str
    required: bool = False
    hints: List[str] = None           # regex “hints” to locate values in messy text
    validate_re: Optional[str] = None # optional validation regex

FIELD_SPECS: List[FieldSpec] = [
    # Identity and contact (Officer manual + OCR scope)
    FieldSpec(name="Full Name", required=True,
              hints=[r"name\s*[:\-]", r"applicant\s*name"]),
    FieldSpec(name="NRIC/FIN", required=True,
              hints=[r"nric\s*/?\s*fin\s*[:\-]", r"nric\s*[:\-]", r"fin\s*[:\-]"],
              validate_re=r"^[STFG]\d{7}[A-Z]$"),
    FieldSpec(name="Citizenship", required=True,
              hints=[r"citizenship\s*[:\-]", r"nationality\s*[:\-]"]),
    FieldSpec(name="Charge Type", required=False,
              hints=[r"charge\s*type\s*[:\-]", r"(charge|offen[cs]e)\s*[:\-]"]),
    FieldSpec(name="Contact Number", required=True,
              hints=[r"(contact|phone|mobile|hp|handphone)\s*(no\.?|number)?\s*[:\-]"],
              validate_re=r"^\+?\d[\d\s\-]{6,}"),
    FieldSpec(name="Address", required=False,
              hints=[r"address\s*[:\-]"]),

    # --- Additional requested fields for OCR upload ---
    FieldSpec(name="Occupation", required=False,
              hints=[r"occupation\s*[:\-]", r"employment\s*type\s*[:\-]", r"job\s*title\s*[:\-]"]),
    FieldSpec(name="Gross Monthly Income", required=False,
              hints=[r"gross\s*monthly\s*income\s*[:\-]", r"monthly\s*income\s*[:\-]", r"income\s*\(monthly\)\s*[:\-]"],
              validate_re=r"^\$?\s?\d[\d,\.]*\s*(sgd)?$"),
    FieldSpec(name="Period of Employment", required=False,
              hints=[r"period\s*of\s*employment\s*[:\-]", r"employment\s*period\s*[:\-]", r"duration\s*of\s*employment\s*[:\-]"]),
    FieldSpec(name="Family member Name", required=False,
              hints=[r"family\s*member\s*name\s*[:\-]", r"household\s*member\s*name\s*[:\-]"]),
    FieldSpec(name="Relationship", required=False,
              hints=[r"relationship\s*[:\-]"]),
    FieldSpec(name="Family members Occupation", required=False,
              hints=[r"family\s*members?\s*occupation\s*[:\-]", r"household\s*member\s*occupation\s*[:\-]"]),
    FieldSpec(name="Family members Gross Monthly Income", required=False,
              hints=[r"family\s*members?\s*gross\s*monthly\s*income\s*[:\-]", r"household\s*member\s*monthly\s*income\s*[:\-]"],
              validate_re=r"^\$?\s?\d[\d,\.]*\s*(sgd)?$"),
]

# Storage for extracted cases during the demo
extracted_cases: List[Dict[str, Any]] = []
EASY_OCR_READER = None  # lazy-initialized

def _normalise_value(raw: str) -> str:
    v = raw.strip().lower()
    return NORMALISE_MAP.get(v, raw.strip())

def _get_easyocr_reader():
    global EASY_OCR_READER
    if EASY_OCR_READER is None and EASY_OCR_AVAILABLE:
        # English only by default; CPU mode for portability
        EASY_OCR_READER = easyocr.Reader(["en"], gpu=False)
    return EASY_OCR_READER


def _ocr_image_to_text(file_stream: io.BytesIO, engine: Optional[str] = None) -> str:
    """Best-effort OCR.
    - If engine == 'easyocr' and EasyOCR is available, try EasyOCR (better for handwriting)
    - Else try Tesseract via PIL
    - Else fallback: try decode as UTF-8 plaintext
    """
    # Prefer EasyOCR if requested
    if engine == "easyocr" and EASY_OCR_AVAILABLE:
        try:
            if not OCR_AVAILABLE:
                raise RuntimeError("PIL not available for image decode")
            file_stream.seek(0)
            image = Image.open(file_stream).convert("RGB")
            import numpy as np  # EasyOCR depends on numpy; import lazily
            arr = np.array(image)
            reader = _get_easyocr_reader()
            if reader is not None:
                lines = reader.readtext(arr, detail=0, paragraph=True)
                if lines:
                    return "\n".join(lines)
        except Exception:
            pass

    # Default: Tesseract
    if OCR_AVAILABLE:
        try:
            file_stream.seek(0)
            image = Image.open(file_stream)
            return pytesseract.image_to_string(image)
        except Exception:
            pass
    # Fallback – treat as plaintext upload or return empty
    try:
        content = file_stream.getvalue().decode("utf-8", errors="ignore")
        if content.strip():
            return content
    except Exception:
        pass
    return ""

def _find_field_value(text: str, spec: FieldSpec) -> Optional[str]:
    # Heuristic 1: search for hint lines; take content after ':' or '-'
    if spec.hints:
        for hint in spec.hints:
            m = re.search(hint, text, flags=re.IGNORECASE)
            if m:
                start = text.rfind('\n', 0, m.start()) + 1
                end = text.find('\n', m.end())
                if end == -1:
                    end = len(text)
                line = text[start:end]
                parts = re.split(r"[:\-]", line, maxsplit=1)
                if len(parts) == 2:
                    candidate = parts[1].strip()
                    if candidate:
                        return candidate
    # Heuristic 2: global patterns for some fields
    if spec.name.startswith("NRIC"):
        m = re.search(r"[STFG]\d{7}[A-Z]", text, flags=re.IGNORECASE)
        if m:
            return m.group(0).upper()
    if spec.name == "Contact Number":
        m = re.search(r"\+?\d[\d\s\-]{6,}\d", text)
        if m:
            return re.sub(r"\s+", "", m.group(0))
    return None

def _validate(spec: FieldSpec, value: str) -> bool:
    if not value:
        return False
    if spec.validate_re:
        return re.match(spec.validate_re, value.strip(), flags=re.IGNORECASE) is not None
    return True

def parse_text_to_structured(text: str) -> Dict[str, Any]:
    fields: Dict[str, Dict[str, Any]] = {}
    missing_required: List[str] = []
    flags: List[str] = []

    for spec in FIELD_SPECS:
        raw = _find_field_value(text, spec)
        if raw is None or not raw.strip():
            if spec.required:
                missing_required.append(spec.name)
            value = "Not Provided"
            valid = False
        else:
            value = _normalise_value(raw)
            valid = _validate(spec, value)
            if not valid:
                flags.append(f"Validation failed for {spec.name}")
            if value in ("Unknown", "Not Provided"):
                flags.append(f"Ambiguous {spec.name} -> '{value}'")
        fields[spec.name] = {"value": value, "valid": valid}

    total = len(FIELD_SPECS)
    confidence = round(100.0 * sum(1 for f in fields.values() if f["valid"]) / max(total, 1), 1)

    return {
        "fields": fields,
        "missing_required": missing_required,
        "flags": flags,
        "confidence": confidence,
        "raw_text": text,
    }

# ----------------- End PDO OCR + Parsing Layer (Prototype) -----------------

reports = [
    {"id": "1", "date": "2024-08-08", "genres": ["spam", "nudity"], "post_owner": "User123", "status": "Complete"},
    {"id": "2", "date": "2024-08-09", "genres": ["hate"], "post_owner": "User456", "status": "Reviewing"},
]

# ---- Simple role-based access (demo) ----
def current_role() -> str:
    return session.get("role", "")


def requires_role(*roles):
    def deco(f):
        @wraps(f)
        def wrapper(*a, **kw):
            role = current_role()
            if role in roles:
                return f(*a, **kw)
            # If JSON or POST, return 403; otherwise redirect to login
            wants_json = request.is_json or request.path.endswith("/decision") or request.method == "POST"
            if wants_json:
                return jsonify({"error": "Unauthorized"}), 403
            return redirect(url_for("login", next=request.path))
        return wrapper
    return deco


@app.context_processor
def inject_role():
    return {"role": current_role()}


@app.route("/login", methods=["GET", "POST"])
def login():
    officer_pw = os.environ.get("OFFICER_PASSWORD", "officer123")
    manager_pw = os.environ.get("MANAGER_PASSWORD", "manager123")
    if request.method == "POST":
        role = (request.form.get("role") or "").strip().lower()
        pw = request.form.get("password") or ""
        if role == "officer" and pw == officer_pw:
            session["role"] = "officer"
        elif role == "manager" and pw == manager_pw:
            session["role"] = "manager"
        else:
            # invalid
            return render_template("login.html", error="Invalid role/password", officer_pw=officer_pw, manager_pw=manager_pw), 401
        # Default landing per role
        default_next = url_for("manager_dashboard") if session.get("role") == "manager" else url_for("home")
        nxt = request.form.get("next") or request.args.get("next") or default_next
        return redirect(nxt)
    return render_template("login.html", officer_pw=officer_pw, manager_pw=manager_pw)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))

# Upload config for demo (do not use in production without authentication!)
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "tiff", "tif", "bmp", "gif", "pdf", "txt"}
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def home():
    if not session.get("role"):
        return redirect(url_for("login"))
    return render_template("index.html")

@app.route("/profile")
def profile():
    return render_template("profile.html", reports=reports)

@app.route("/submit_report", methods=["POST"])
def submit_report():
    data = request.get_json()
    new_report = {
        "id": str(len(reports) + 1),
        "date": datetime.now().strftime("%Y-%m-%d"),
        "genres": data["genres"],
        "post_owner": "UserXYZ",
        "status": "Submitted"
    }
    reports.append(new_report)
    return jsonify({"message": "Report submitted successfully"})

# ---- NEW: OCR/Parsing endpoints ----
@app.route("/ocr/upload", methods=["POST"])
@requires_role("officer")
def ocr_upload():
    """Accepts one or more files and returns extracted cases.
    - Multiple files: send as form field 'files' (can include several)
    - Single file: legacy 'file' still supported
    - Optional: 'engine' form field (e.g., 'easyocr')
    """
    files = []
    if "files" in request.files:
        files = [f for f in request.files.getlist("files") if f and f.filename]
    elif "file" in request.files:
        f = request.files["file"]
        if f and f.filename:
            files = [f]

    if not files:
        return jsonify({"error": "No file(s) provided"}), 400

    engine = request.form.get("engine") or request.args.get("engine")
    created = []
    for f in files:
        if f.filename == "":
            continue
        if not allowed_file(f.filename):
            created.append({"filename": f.filename, "error": "Unsupported file type"})
            continue
        filename = secure_filename(f.filename)
        buf = io.BytesIO(f.read())
        text = _ocr_image_to_text(buf, engine=engine)
        structured = parse_text_to_structured(text)
        case = {
            "case_id": str(len(extracted_cases) + 1),
            "filename": filename,
            "extracted": structured,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "decision": "Pending",
        }
        extracted_cases.append(case)
        created.append(case)

    if len(created) == 1 and "error" not in created[0]:
        return jsonify(created[0])
    return jsonify({"count": len(created), "cases": created})


@app.route("/manual/create", methods=["POST"])
@requires_role("officer")
def manual_create():
    """Create a case from manual officer input.
    Accepts JSON (preferred) with a mapping of field names to values, e.g.:
    {"Full Name": "Jane", "NRIC/FIN": "S1234567A", ...}
    Also supports form-encoded input using the same field names.
    """
    # Gather incoming values (JSON or form)
    incoming = {}
    if request.is_json:
        incoming = request.get_json(silent=True) or {}
    else:
        # Use exact field names as keys if present; also allow simplified keys
        incoming = {k: v for k, v in request.form.items()}

    fields: Dict[str, Dict[str, Any]] = {}
    missing_required: List[str] = []
    flags: List[str] = []

    for spec in FIELD_SPECS:
        # Accept exact field key or a simplified key (lowercase, underscores)
        raw = incoming.get(spec.name)
        if raw is None:
            simple_key = spec.name.lower().replace("/", "_").replace(" ", "_")
            raw = incoming.get(simple_key)
        if raw is None or not str(raw).strip():
            if spec.required:
                missing_required.append(spec.name)
            value = "Not Provided"
            valid = False
        else:
            value = _normalise_value(str(raw))
            valid = _validate(spec, value)
            if not valid:
                flags.append(f"Validation failed for {spec.name}")
            if value in ("Unknown", "Not Provided"):
                flags.append(f"Ambiguous {spec.name} -> '{value}'")
        fields[spec.name] = {"value": value, "valid": valid}

    total = len(FIELD_SPECS)
    confidence = round(100.0 * sum(1 for f in fields.values() if f["valid"]) / max(total, 1), 1)

    structured = {
        "fields": fields,
        "missing_required": missing_required,
        "flags": flags,
        "confidence": confidence,
        "raw_text": "(manual entry)",
    }

    case = {
        "case_id": str(len(extracted_cases) + 1),
        "filename": "manual-entry",
        "extracted": structured,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "decision": "Pending",
    }
    extracted_cases.append(case)
    return jsonify(case)

@app.route("/cases", methods=["GET"])
def list_cases():
    return jsonify({"count": len(extracted_cases), "cases": extracted_cases})

@app.route("/cases/view", methods=["GET"])
def cases_view():
    # Render a simple list of cases with links
    items = sorted(extracted_cases, key=lambda c: int(c["case_id"]), reverse=True)
    return render_template("cases.html", cases=items)

@app.route("/cases/<case_id>", methods=["GET"])
def get_case(case_id: str):
    case = next((c for c in extracted_cases if c["case_id"] == case_id), None)
    if not case:
        return jsonify({"error": "Case not found"}), 404
    return jsonify(case)

@app.route("/case/<case_id>", methods=["GET"])
def view_case(case_id: str):
    case = next((c for c in extracted_cases if c["case_id"] == case_id), None)
    if not case:
        return render_template("case.html", case=None, error="Case not found"), 404
    return render_template("case.html", case=case)

# ---- Summary (Officer) ----
@app.route("/summary", methods=["GET"])
@requires_role("officer")
def summary_view():
    # Build table headers from active FIELD_SPECS
    headers = [fs.name for fs in FIELD_SPECS]
    # Prepare rows from extracted_cases
    items = []
    for c in sorted(extracted_cases, key=lambda x: int(x["case_id"])):
        row = {
            "case_id": c.get("case_id"),
            "status": c.get("decision", "Pending"),
            "fields": {h: (c.get("extracted", {}).get("fields", {}).get(h, {}).get("value", "")) for h in headers},
        }
        items.append(row)
    return render_template("summary.html", headers=headers, items=items)

@app.route("/case/<case_id>/decision", methods=["POST"])
@requires_role("manager")
def set_case_decision(case_id: str):
    case = next((c for c in extracted_cases if c["case_id"] == case_id), None)
    if not case:
        return jsonify({"error": "Case not found"}), 404
    data = request.get_json(silent=True) or {}
    status = (data.get("status") or "").strip().lower()
    if status not in {"approved", "declined"}:
        return jsonify({"error": "Invalid status. Use 'approved' or 'declined'."}), 400
    decision = "Approved" if status == "approved" else "Declined"
    case["decision"] = decision
    case["decision_at"] = datetime.now().isoformat(timespec="seconds")
    return jsonify({"ok": True, "case": case})

@app.route("/cases/processed/upload", methods=["POST"])
@requires_role("officer")
def upload_processed_cases():
    """Upload a CSV mapping case decisions.
    Expected CSV headers: case_id,status  (status in {approved, declined})
    Returns summary with updated cases and any errors.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file part in request"}), 400
    f = request.files["file"]
    if f.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # Read CSV content
    try:
        content = f.read().decode("utf-8-sig", errors="ignore")
    except Exception as e:
        return jsonify({"error": f"Failed to read file: {e}"}), 400

    reader = csv.DictReader(io.StringIO(content))
    required = {"case_id", "status"}
    if not required.issubset({(h or '').strip().lower() for h in reader.fieldnames or []}):
        return jsonify({"error": "CSV must have headers: case_id,status"}), 400

    normalize_status = lambda s: (s or "").strip().lower()
    updated = []
    not_found = []
    invalid_status = []

    # Build index for faster lookup
    index = {c["case_id"]: c for c in extracted_cases}

    for row in reader:
        case_id = (row.get("case_id") or row.get("CASE_ID") or "").strip()
        status = normalize_status(row.get("status") or row.get("STATUS"))
        if not case_id:
            continue
        if status not in {"approved", "declined"}:
            invalid_status.append({"case_id": case_id, "status": row.get("status")})
            continue
        case = index.get(case_id)
        if not case:
            not_found.append(case_id)
            continue
        decision = "Approved" if status == "approved" else "Declined"
        case["decision"] = decision
        case["decision_at"] = datetime.now().isoformat(timespec="seconds")
        updated.append({"case_id": case_id, "decision": decision, "filename": case.get("filename"), "created_at": case.get("created_at")})

    return jsonify({
        "updated_count": len(updated),
        "updated": updated,
        "not_found": not_found,
        "invalid_status": invalid_status,
    })

@app.route("/export/csv", methods=["GET"])
def export_csv():
    if not extracted_cases:
        return jsonify({"error": "No cases to export"}), 400

    headers = [fs.name for fs in FIELD_SPECS]
    # Exclude sensitive/PII fields from CSV export as requested
    exclude = {"Full Name", "NRIC/FIN", "Charge Type"}
    export_headers = [h for h in headers if h not in exclude]

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=["case_id", "filename"] + export_headers)
    writer.writeheader()

    for case in extracted_cases:
        row = {"case_id": case["case_id"], "filename": case["filename"]}
        fields = case["extracted"]["fields"]
        for h in export_headers:
            row[h] = fields.get(h, {}).get("value", "")
        writer.writerow(row)

    mem = io.BytesIO(output.getvalue().encode("utf-8"))
    mem.seek(0)
    return send_file(mem, mimetype="text/csv", as_attachment=True, download_name="pdo_cases.csv")

@app.route("/map-to-db", methods=["POST"])
def map_to_db():
    payload = []
    for case in extracted_cases:
        record = {"case_id": case["case_id"], "created_at": case["created_at"]}
        for k, v in case["extracted"]["fields"].items():
            record[k] = v.get("value")
        record["_flags"] = case["extracted"].get("flags", [])
        record["_missing_required"] = case["extracted"].get("missing_required", [])
        record["_confidence"] = case["extracted"].get("confidence", 0)
        payload.append(record)
    return jsonify({"records": payload})

@app.route("/ai_suggestion", methods=["POST"])
def ai_suggestion():
    data = request.get_json()
    text = data.get("text", "")
    suggested_genre = find_genre_from_text(text)
    return jsonify({"suggested_genre": suggested_genre})

@app.route("/check_report/<report_id>")
def check_report(report_id):
    report = next((r for r in reports if r["id"] == report_id), None)
    if report and report["status"] == "Complete":
        return jsonify({"success": True, "message": "The post has been taken down."})
    return jsonify({"success": False, "message": "The report is not complete or not found."})

def find_genre_from_text(text):
    text = text.lower()
    for genre, keywords in suggestion_map.items():
        if any(keyword in text for keyword in keywords):
            return genre
    return None

# Manager dashboard route
@app.route("/manager")
@requires_role("manager")
def manager_dashboard():
    return render_template("manager.html")



if __name__ == "__main__":
    app.run(debug=True)
