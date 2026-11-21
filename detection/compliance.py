"""
SiteGuard AI - Dedicated PPE Compliance Evaluation Module
Supports multi-PPE detection:
  - Hardhat / Helmet & NO-Hardhat / no_helmet
  - Safety Vest & NO-Safety Vest
  - Goggles & NO-Goggles / no_goggle (NEW)
  - Gloves & NO-Gloves / no_gloves (NEW)
  - Boots & NO-Boots / no_boots (NEW)
  - Mask & NO-Mask
Uses spatial body-part association and labels status strictly as:
  COMPLIANT, NON-COMPLIANT, or UNVERIFIED.
"""

def box_center(box):
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)

def point_in_box(px, py, box, margin_x=0.08, margin_y=0.08):
    bw = box[2] - box[0]
    bh = box[3] - box[1]
    return (box[0] - margin_x * bw <= px <= box[2] + margin_x * bw and
            box[1] - margin_y * bh <= py <= box[3] + margin_y * bh)

def intersection_ratio(boxA, boxB):
    """Calculates intersection area relative to smaller box area."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxB[3], boxB[3])

    interW = max(0.0, xB - xA)
    interH = max(0.0, yB - yA)
    interArea = interW * interH

    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    if areaB <= 0:
        return 0.0
    return interArea / areaB

class ComplianceEngine:
    def __init__(self):
        pass

    def evaluate(self, detections, tracked_workers=None):
        """
        Evaluates safety compliance for all detected workers in a single frame/image.
        """
        # Separate workers and other detections
        person_detections = [
            d for d in detections 
            if d['name'].lower() in ['person', 'worker']
        ]

        # Categorize detected items by normalized name first so fallbacks can use them
        helmets, no_helmets = [], []
        vests, no_vests = [], []
        goggles, no_goggles = [], []
        gloves, no_gloves = [], []
        boots, no_boots = [], []
        masks, no_masks = [], []

        for d in detections:
            cname = d['name'].lower()
            if cname in ['helmet', 'hardhat']:
                helmets.append(d)
            elif cname in ['no_helmet', 'no-hardhat', 'no_hardhat']:
                no_helmets.append(d)
            elif cname in ['vest', 'safety vest', 'safety_vest']:
                vests.append(d)
            elif cname in ['no_vest', 'no-safety vest', 'no_safety_vest', 'no-vest']:
                no_vests.append(d)
            elif cname in ['goggles', 'goggle']:
                goggles.append(d)
            elif cname in ['no_goggle', 'no-goggles', 'no_goggles', 'no-goggle']:
                no_goggles.append(d)
            elif cname in ['gloves', 'glove']:
                gloves.append(d)
            elif cname in ['no_gloves', 'no-gloves', 'no_glove']:
                no_gloves.append(d)
            elif cname in ['boots', 'boot']:
                boots.append(d)
            elif cname in ['no_boots', 'no-boots', 'no_boot']:
                no_boots.append(d)
            elif cname in ['mask']:
                masks.append(d)
            elif cname in ['no-mask', 'no_mask']:
                no_masks.append(d)

        # Build worker candidate list
        workers_list = []
        if tracked_workers:
            for tw in tracked_workers:
                matching_conf = 0.85
                for pd in person_detections:
                    if intersection_ratio(tw['box'], pd['box']) > 0.4:
                        matching_conf = pd['conf']
                        break
                workers_list.append({
                    'worker_id': tw['id'],
                    'box': tw['box'],
                    'conf': matching_conf
                })
        else:
            for idx, pd in enumerate(person_detections):
                workers_list.append({
                    'worker_id': idx + 1,
                    'box': pd['box'],
                    'conf': pd['conf']
                })

        # Fallback for close-up webcam feeds: if no person box was detected, but PPE/NO-PPE was found
        if len(workers_list) == 0:
            primary_items = no_helmets + helmets + no_vests + vests
            for idx, ppe in enumerate(primary_items[:2]):
                pbox = ppe['box']
                pw = pbox[2] - pbox[0]
                ph = pbox[3] - pbox[1]
                inferred_box = [
                    max(0.0, pbox[0] - pw * 0.4),
                    max(0.0, pbox[1] - ph * 0.2),
                    pbox[2] + pw * 0.4,
                    pbox[3] + ph * 3.5
                ]
                workers_list.append({
                    'worker_id': idx + 1,
                    'box': inferred_box,
                    'conf': ppe['conf']
                })

        evaluated_workers = []
        frame_violations = []

        compliant_count = 0
        non_compliant_count = 0
        unverified_count = 0

        for w in workers_list:
            wx1, wy1, wx2, wy2 = w['box']
            w_w = wx2 - wx1
            w_h = wy2 - wy1

            # Geometric body regions
            head_box = [wx1 - 0.05 * w_w, wy1 - 0.15 * w_h, wx2 + 0.05 * w_w, wy1 + 0.35 * w_h]
            eye_box = [wx1 - 0.05 * w_w, wy1 + 0.05 * w_h, wx2 + 0.05 * w_w, wy1 + 0.30 * w_h]
            torso_box = [wx1 - 0.05 * w_w, wy1 + 0.20 * w_h, wx2 + 0.05 * w_w, wy1 + 0.75 * w_h]
            hand_box = [wx1 - 0.12 * w_w, wy1 + 0.40 * w_h, wx2 + 0.12 * w_w, wy2 + 0.05 * w_h]
            foot_box = [wx1 - 0.08 * w_w, wy1 + 0.75 * w_h, wx2 + 0.08 * w_w, wy2 + 0.15 * w_h]

            missing_items = []
            detected_items = []

            # 1. HELMET EVALUATION
            helmet_status = "NOT_DETECTED"
            helmet_conf = 0.0
            for nh in no_helmets:
                cx, cy = box_center(nh['box'])
                if point_in_box(cx, cy, head_box) or intersection_ratio(head_box, nh['box']) > 0.20:
                    helmet_status = "VIOLATION"
                    helmet_conf = nh['conf']
                    missing_items.append("Missing Helmet")
                    break
            if helmet_status != "VIOLATION":
                for h in helmets:
                    cx, cy = box_center(h['box'])
                    if point_in_box(cx, cy, head_box) or intersection_ratio(head_box, h['box']) > 0.20:
                        helmet_status = "WORN"
                        helmet_conf = h['conf']
                        detected_items.append("Helmet")
                        break

            # If neither positive nor explicit negative was detected on a clearly visible worker:
            if helmet_status == "NOT_DETECTED":
                if w_h >= 60 and w_w >= 25:
                    helmet_status = "VIOLATION"
                    missing_items.append("Missing Helmet")
                else:
                    helmet_status = "UNVERIFIED"

            # 2. VEST EVALUATION
            vest_status = "NOT_DETECTED"
            vest_conf = 0.0
            for nv in no_vests:
                cx, cy = box_center(nv['box'])
                if point_in_box(cx, cy, torso_box) or intersection_ratio(torso_box, nv['box']) > 0.20:
                    vest_status = "VIOLATION"
                    vest_conf = nv['conf']
                    missing_items.append("Missing Safety Vest")
                    break
            if vest_status != "VIOLATION":
                for v in vests:
                    cx, cy = box_center(v['box'])
                    if point_in_box(cx, cy, torso_box) or intersection_ratio(torso_box, v['box']) > 0.20:
                        vest_status = "WORN"
                        vest_conf = v['conf']
                        detected_items.append("Safety Vest")
                        break

            # If neither positive nor explicit negative was detected on a clearly visible worker:
            if vest_status == "NOT_DETECTED":
                if w_h >= 75 and w_w >= 30:
                    vest_status = "VIOLATION"
                    missing_items.append("Missing Safety Vest")
                else:
                    vest_status = "UNVERIFIED"

            # 3. GOGGLES EVALUATION
            goggles_status = "NOT_DETECTED"
            for ng in no_goggles:
                cx, cy = box_center(ng['box'])
                if point_in_box(cx, cy, eye_box) or intersection_ratio(eye_box, ng['box']) > 0.25:
                    goggles_status = "VIOLATION"
                    missing_items.append("Missing Goggles")
                    break
            if goggles_status != "VIOLATION":
                for g in goggles:
                    cx, cy = box_center(g['box'])
                    if point_in_box(cx, cy, eye_box) or intersection_ratio(eye_box, g['box']) > 0.2:
                        goggles_status = "WORN"
                        detected_items.append("Goggles")
                        break

            # 4. GLOVES EVALUATION
            gloves_status = "NOT_DETECTED"
            for ng in no_gloves:
                cx, cy = box_center(ng['box'])
                if point_in_box(cx, cy, hand_box) or intersection_ratio(hand_box, ng['box']) > 0.2:
                    gloves_status = "VIOLATION"
                    missing_items.append("Missing Gloves")
                    break
            if gloves_status != "VIOLATION":
                for gl in gloves:
                    cx, cy = box_center(gl['box'])
                    if point_in_box(cx, cy, hand_box) or intersection_ratio(hand_box, gl['box']) > 0.2:
                        gloves_status = "WORN"
                        detected_items.append("Gloves")
                        break

            # 5. BOOTS EVALUATION
            boots_status = "NOT_DETECTED"
            for nb in no_boots:
                cx, cy = box_center(nb['box'])
                if point_in_box(cx, cy, foot_box) or intersection_ratio(foot_box, nb['box']) > 0.2:
                    boots_status = "VIOLATION"
                    missing_items.append("Missing Boots")
                    break
            if boots_status != "VIOLATION":
                for b in boots:
                    cx, cy = box_center(b['box'])
                    if point_in_box(cx, cy, foot_box) or intersection_ratio(foot_box, b['box']) > 0.2:
                        boots_status = "WORN"
                        detected_items.append("Boots")
                        break

            # 6. OVERALL COMPLIANCE DECISION
            is_adequate = (w_h >= 65 and w_w >= 30)

            if len(missing_items) > 0:
                overall_status = "NON-COMPLIANT"
                non_compliant_count += 1
                for item in missing_items:
                    frame_violations.append({
                        'worker_id': w['worker_id'],
                        'missing_ppe': item,
                        'confidence': w['conf'],
                        'status': "NON-COMPLIANT"
                    })
            elif helmet_status == "WORN" and vest_status == "WORN":
                overall_status = "COMPLIANT"
                compliant_count += 1
            else:
                overall_status = "UNVERIFIED"
                unverified_count += 1

            evaluated_workers.append({
                'worker_id': w['worker_id'],
                'box': w['box'],
                'conf': w['conf'],
                'helmet_status': helmet_status,
                'vest_status': vest_status,
                'goggles_status': goggles_status,
                'gloves_status': gloves_status,
                'boots_status': boots_status,
                'overall_status': overall_status,
                'missing_items': missing_items,
                'detected_items': detected_items
            })

        violator_ids = [w['worker_id'] for w in evaluated_workers if w['overall_status'] == "NON-COMPLIANT"]
        violator_info = ", #".join(str(vid) for vid in violator_ids)
        if violator_info:
            violator_info = f"#{violator_info}"

        summary = {
            'total_workers': len(workers_list),
            'compliant': compliant_count,
            'non_compliant': non_compliant_count,
            'unverified': unverified_count,
            'violator_ids': violator_ids,
            'violator_info': violator_info
        }

        return {
            'workers': evaluated_workers,
            'summary': summary,
            'violations': frame_violations
        }
