import os, hashlib, time, uuid
from flask import Blueprint, render_template, request, redirect, url_for, flash, session, jsonify
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename
from models.models import db, User, FIR, Evidence, EvidenceAudit
from blockchain.blockchain_utils import record_on_chain
from datetime import datetime
from Trufor_main.run_trufor import run_trufor
from FractalVideoGuard_main.run_video import run_video_analysis

UPLOAD_FOLDER = 'static/uploads/evidence'

controllers = Blueprint('controllers', __name__)

# ------------------- LOGIN -------------------
@controllers.route('/<role>_login', methods=['GET', 'POST'])
def login(role):
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        # Find user by email + department
        user = User.query.filter_by(email=email, department=role).first()

        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id
            session['department'] = user.department
            session['is_admin'] = user.is_admin

            flash('Login successful ✅', 'success')

            record_on_chain(user.department.capitalize(), "Logged In")

            # ADMIN OVERRIDE
            if user.is_admin:
                return redirect(url_for('controllers.admin_dashboard'))

            return redirect(url_for(f'controllers.{role}_dashboard'))

        flash('Invalid credentials ❌', 'danger')

    return render_template(f'{role}_login.html')

# ------------------- DASHBOARDS -------------------

# ------------------- ADMIN DASHBOARD -------------------
@controllers.route('/admin_dashboard')
def admin_dashboard():
    if not session.get('is_admin'):
        return redirect(url_for('controllers.login', role='admin'))

    users = User.query.all()
    return render_template('admin_dashboard.html', users=users)

# ------------------- POLICE DASHBOARD -------------------
@controllers.route('/police_dashboard')
def police_dashboard():
    if 'user_id' not in session or session.get('department') != 'police':
        return redirect(url_for('controllers.login', role='police'))

    firs = FIR.query.filter_by(police_id=session['user_id']).all()

    # ✅ Convert FIR objects to JSON-safe dictionaries
    firs_data = []
    for fir in firs:
        firs_data.append({
            "id": fir.id,
            "fir_number": fir.fir_number,

            "complainant_name": fir.complainant_name,
            "guardian_type": fir.guardian_type,
            "guardian_name": fir.guardian_name,
            "age": fir.age,
            "gender": fir.gender,
            "marital_status": fir.marital_status,
            "occupation": fir.occupation,
            "aadhaar": fir.aadhaar or "N/A",
            "mobile": fir.mobile,
            "address": fir.address,

            "incident_date": fir.incident_date.strftime('%d-%m-%Y'),
            "incident_time": fir.incident_time,
            "location": fir.location,
            "police_station": fir.police_station,
            "offence_nature": fir.offence_nature,
            "sections": fir.sections,

            "accused_known": fir.accused_known,
            "accused_name": fir.accused_name or "N/A",
            "accused_address": fir.accused_address or "N/A",

            "description": fir.description,
            "status": fir.status,
            "created_at": fir.created_at.strftime('%d-%m-%Y')
        })
        # -------- STEP-1 ADD THIS BLOCK --------
        # total evidence uploaded by this police officer
        evidences_count = Evidence.query.filter_by(
            uploaded_by=session['user_id']
        ).count()

        # pending forensic review (basic example)
        pending_forensic_count = Evidence.query.filter_by(
            uploaded_by=session['user_id'],
            blockchain_status="Recorded"
        ).count()
        # FIR counts by status
        active_firs = FIR.query.filter_by(police_id=session['user_id'], status="Active").count()
        under_investigation = FIR.query.filter_by(police_id=session['user_id'], status="Under Investigation").count()
        closed_firs = FIR.query.filter_by(police_id=session['user_id'], status="Closed").count()

        # Recent activities – last 5 evidences
        recent_evidence = Evidence.query.filter_by(uploaded_by=session['user_id']) \
            .order_by(Evidence.uploaded_at.desc()).limit(5).all()

        # Performance KPIs
        total_firs = len(firs)
        solved_rate = int((closed_firs / total_firs) * 100) if total_firs > 0 else 0

        # Blockchain logs preview
        recent_audit = EvidenceAudit.query.order_by(EvidenceAudit.timestamp.desc()).limit(5).all()

    # ================= EVIDENCE HISTORY =================
    evidence_history = (
        db.session.query(Evidence, FIR, User)
        .join(FIR, Evidence.fir_id == FIR.id)
        .join(User, Evidence.uploaded_by == User.id)
        .filter(Evidence.uploaded_by == session['user_id'])
        .order_by(Evidence.uploaded_at.desc())
        .all()
    )
    return render_template(
        'police_dashboard.html',
        firs=firs,
        firs_data=firs_data,
        evidence_history=evidence_history,
        evidences=evidences_count,
        pending_forensic=pending_forensic_count,
        active_firs=active_firs,
        under_investigation=under_investigation,
        closed_firs=closed_firs,
        recent_evidence=recent_evidence,
        solved_rate=solved_rate,
        recent_audit=recent_audit
    )

# ------------------- FORENSIC DASHBOARD -------------------
@controllers.route('/forensic_dashboard')
def forensic_dashboard():
    if 'user_id' not in session or session.get('department') != 'forensic':
        return redirect(url_for('controllers.login', role='forensic'))

    evidences = (
        db.session.query(Evidence, FIR, User)
        .join(FIR, Evidence.fir_id == FIR.id)
        .join(User, Evidence.uploaded_by == User.id)
        .filter(Evidence.forensic_status == "Pending")
        .order_by(Evidence.uploaded_at.desc())
        .all()
    )

    evidence_list = []
    for evidence, fir, user in evidences:
        evidence_list.append({
            "seal_id": evidence.evidence_seal_id,
            "fir_no": fir.fir_number,
            "evidence_type": evidence.evidence_type,
            "uploaded_by_name": user.fullname,
            "uploaded_by_role": user.designation,
            "status": evidence.forensic_status,
            "file_name": evidence.stored_filename   # ✅ IMPORTANT
        })

    return render_template(
        'forensic_dashboard.html',
        evidences=evidence_list,
        evidence_files=[e["file_name"] for e in evidence_list]  # ✅ FOR DROPDOWN
    )


# ------------------- FORENSIC TOOLS - TRUFOR ANALYSIS -------------------
@controllers.route('/forensic/run_trufor/<filename>')
def run_trufor_analysis(filename):

    image_path = os.path.join("static/uploads/evidence", filename)
    output_dir = "static/trufor_output"

    run_trufor(image_path, output_dir)

    npz_file = os.path.join(
        output_dir,
        filename + ".npz"
    )

    png_file = os.path.join(
        output_dir,
        filename + "_heatmap.png"
    )

    from utils.npz_to_png import npz_to_outputs

    base = os.path.splitext(filename)[0]

    results = npz_to_outputs(
        npz_path=npz_file,
        original_img_path=image_path,
        output_dir=output_dir,
        base_name=base
    )

    return {
    "original": f"/static/uploads/evidence/{filename}",
    "heatmap": f"/static/trufor_output/{base}_heatmap.png",
    "overlay": f"/static/trufor_output/{base}_overlay.png",
    "score": results["score"],
    "verdict": results["verdict"],
    "risk": results["risk"],
    # 🔹 NEW
    "metrics": results["metrics"],
    "exif": results["exif"]
}

# ------------------- FORENSIC TOOLS - TRUFOR DROPDOWN -------------------
@controllers.route('/forensic_tools')
def forensic_tools():

    evidence_dir = "static/uploads/evidence"
    evidence_files = [
        f for f in os.listdir(evidence_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]

    return render_template(
        "forensic_tools.html",
        evidence_files=evidence_files
    )

# ------------------- FORENSIC TOOLS - Video Analysis Tool -------------------

@controllers.route('/forensic/run_video/<filename>')
def run_video(filename):
    try:
        BASE_DIR = os.path.abspath(os.getcwd())

        video_path = os.path.join(
            BASE_DIR, "static", "uploads", "evidence", filename
        )

        output_json = os.path.join(
            BASE_DIR, "static", "video_output",
            os.path.splitext(filename)[0],
            "analysis.json"
        )

        print("🎥 Running video analysis:", video_path)

        data = run_video_analysis(video_path, output_json)
        features = data.get("features", {})

        # -------- SAFE FEATURE EXTRACTION --------
        fd = float(features.get("fractal_dim_box_mean", 0))
        ring = float(features.get("ringing_mean", 0))
        block = float(features.get("blockiness_mean", 0))

        # -------- VERDICT LOGIC (3 STATES ONLY) --------
        if fd < 1.25 and ring > 4.5:
            verdict = "Likely Manipulated"
            risk = "High"
            reason = [
                "Significantly reduced fractal complexity indicates smoothing or synthesis",
                "Strong ringing artifacts suggest resampling or AI-based processing"
            ]

        elif fd < 1.40 or ring > 3.5:
            verdict = "Suspicious"
            risk = "Medium"
            reason = [
                "Moderate loss of natural texture complexity detected",
                "Artifacts consistent with compression or enhancement pipelines"
            ]

        else:
            verdict = "Likely Authentic"
            risk = "Low"
            reason = [
                "Natural texture complexity preserved across frames",
                "No abnormal compression or resampling artifacts detected"
            ]

        return jsonify({
            "verdict": verdict,
            "risk": risk,
            "reason": reason,        # ✅ Used by frontend
            "features": features,
            "frames": data.get("frames", []),
            "heatmaps": data.get("heatmaps", []),
            "timeline": data.get("timeline", [])
        })

    except Exception as e:
        print("❌ Video analysis error:", str(e))
        return jsonify({
            "error": "Video analysis failed",
            "details": str(e)
        }), 500

# ------------------- ADMIN: ADD USER -------------------
@controllers.route('/admin/add_user', methods=['POST'])
def add_user():
    if not session.get('is_admin'):
        flash('Access denied ❌', 'danger')
        return redirect(url_for('controllers.admin_dashboard'))

    fullname = request.form.get('fullname')
    email = request.form.get('email')
    password = generate_password_hash(request.form.get('password'), method='pbkdf2:sha256')
    department = request.form.get('department')
    designation = request.form.get('designation')

    # AUTO ADMIN FOR SENIOR OFFICERS
    admin_designations = [
        'Superintendent of Police (SP)',
        'Senior Superintendent of Police (SSP)',
        'Director General of Police (DGP)',
        'District & Sessions Judge',
        'Registrar General (High Court)',
        'Director, Forensic Science Laboratory (FSL)'
    ]

    is_admin = designation in admin_designations

    new_user = User(
        fullname=fullname,
        email=email,
        password=password,
        department=department,
        designation=designation,
        is_admin=is_admin
    )

    db.session.add(new_user)
    db.session.commit()

    record_on_chain("Admin", f"Added user: {fullname} ({designation})")

    flash('User added successfully ✅', 'success')
    return redirect(url_for('controllers.admin_dashboard'))

# ------------------- ADMIN: UPDATE ADMIN ACCESS -------------------
@controllers.route('/admin/update_admin/<int:user_id>', methods=['POST'])
def update_admin(user_id):
    if not session.get('is_admin'):
        flash('Access denied ❌', 'danger')
        return redirect(url_for('controllers.admin_dashboard'))

    user = User.query.get(user_id)
    if user:
        user.is_admin = not user.is_admin
        db.session.commit()
        record_on_chain("Admin", f"Admin access toggled for {user.fullname}")
        flash('Admin access updated', 'success')

    return redirect(url_for('controllers.admin_dashboard'))

# ------------------- REGISTER FIR AND VIEW  IT -------------------

@controllers.route('/police/register_fir', methods=['GET', 'POST'])
def register_fir():
    if session.get('department') != 'police':
        return redirect(url_for('controllers.login', role='police'))

    if request.method == 'POST':
        fir_number = f"FIR-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"

        new_fir = FIR(
            fir_number=fir_number,

            complainant_name=request.form['complainant_name'],
            guardian_type=request.form['guardian_type'],
            guardian_name=request.form['guardian_name'],
            age=request.form['age'],
            gender=request.form['gender'],
            marital_status=request.form['marital_status'],
            occupation=request.form['occupation'],
            aadhaar=request.form.get('aadhaar'),
            mobile=request.form['mobile'],
            address=request.form['address'],

            incident_date=datetime.strptime(request.form['incident_date'], "%Y-%m-%d"),
            incident_time=request.form['incident_time'],
            location=request.form['location'],
            police_station=request.form['police_station'],
            offence_nature=request.form['offence_nature'],
            sections=request.form['sections'],

            accused_known=request.form['accused_known'],
            accused_name=request.form.get('accused_name'),
            accused_address=request.form.get('accused_address'),

            description=request.form['description'],
            police_id=session['user_id'],
            status="Active"
        )

        db.session.add(new_fir)
        db.session.commit()

        record_on_chain("Police", f"FIR Registered: {fir_number}")

        flash('FIR registered successfully ✅', 'success')
        return redirect(url_for('controllers.police_dashboard'))

    # GET request fallback (not used but safe)
    return redirect(url_for('controllers.police_dashboard'))

@controllers.route('/police/fir/<int:fir_id>')
def view_fir(fir_id):
    if session.get('department') != 'police':
        return redirect(url_for('controllers.login', role='police'))

    fir = FIR.query.get_or_404(fir_id)
    evidences = Evidence.query.filter_by(fir_id=fir_id).all()

    return render_template(
        'police_dashboard.html',
        fir=fir,
        evidences=evidences
    )

# ------------------- UPLOAD EVIDENCE -------------------

@controllers.route('/police/upload_evidence', methods=['POST'])
def upload_evidence():

    if session.get('department') != 'police':
        return redirect(url_for('controllers.login', role='police'))

    file = request.files.get('evidence_file')
    if not file or file.filename == '':
        flash('No evidence file selected', 'error')
        return redirect(url_for('controllers.police_dashboard'))

    # 🔐 Auto-generated Seal ID (Backend Only)
    evidence_seal_id = f"SEAL-{datetime.utcnow().strftime('%Y%m%d')}-{uuid.uuid4().hex[:8].upper()}"

    # 📄 Form Data
    fir_id = request.form['fir_id']
    title = request.form['title']
    evidence_type = request.form['evidence_type']
    description = request.form['description']
    facility_name = request.form['facility_name']
    collection_room = request.form['collection_room']
    storage_type = request.form['storage_type']
    storage_unit = request.form.get('storage_unit')
    storage_slot = request.form.get('storage_slot')
    sensitivity = request.form.get('sensitivity', 'Normal')

    # 📂 File Handling
    original_filename = file.filename
    stored_filename = f"{int(time.time())}_{secure_filename(original_filename)}"
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    file_path = os.path.join(UPLOAD_FOLDER, stored_filename)
    file.save(file_path)

    # 🔐 Hash
    with open(file_path, 'rb') as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()

    # 🧾 Save Evidence
    evidence = Evidence(
        fir_id=fir_id,
        uploaded_by=session['user_id'],
        title=title,
        evidence_type=evidence_type,
        description=description,
        facility_name=facility_name,
        collection_room=collection_room,
        storage_type=storage_type,
        storage_unit=storage_unit,
        storage_slot=storage_slot,
        evidence_seal_id=evidence_seal_id,
        sensitivity=sensitivity,
        original_filename=original_filename,
        stored_filename=stored_filename,
        file_path=file_path,
        file_size=os.path.getsize(file_path),
        mime_type=file.mimetype,
        file_hash=file_hash,
        upload_ip=request.remote_addr,
        user_agent=request.headers.get('User-Agent')
    )

    db.session.add(evidence)
    db.session.commit()

    # 🧾 Audit
    db.session.add(EvidenceAudit(
        evidence_id=evidence.id,
        action=f"Evidence Uploaded | SealID {evidence_seal_id}",
        actor_id=session['user_id'],
        ip_address=request.remote_addr,
        user_agent=request.headers.get('User-Agent')
    ))

    # 🔗 Blockchain
    tx_hash = record_on_chain(
        "Police",
        f"Evidence Sealed | FIR:{fir_id} | Seal:{evidence_seal_id} | Hash:{file_hash}"
    )

    if tx_hash:
        evidence.blockchain_status = "Recorded"

    db.session.commit()

    flash("Evidence secured with blockchain-backed chain of custody ✅", "success")
    return redirect(url_for('controllers.police_dashboard'))

# ------------------- LOGOUT -------------------
@controllers.route('/logout')
def logout():
    role = session.get('department', 'Unknown')
    session.clear()
    record_on_chain(role.capitalize(), "Logged Out")
    flash('Logged out successfully', 'info')
    return redirect(url_for('home'))
