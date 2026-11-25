"""
Frontend UI Components
Provides styled UI cards, state badges, and metric displays for the Streamlit dashboard.
Strictly based on the project synopsis requirements.
"""

import streamlit as st
from config import settings


def render_header():
    """Renders the top academic title and project context."""
    st.markdown(
        """
        <div style="background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); padding: 22px; border-radius: 10px; margin-bottom: 20px; color: white;">
            <h2 style="margin: 0; font-size: 24px; font-weight: 700; color: #ffffff;">
                AI-Powered Driver Drowsiness Detection System
            </h2>
            <p style="margin: 5px 0 0 0; font-size: 14px; color: #e0e8f5;">
                Computer Vision and Machine Learning for Real-Time Driver Alertness Monitoring and Accident Prevention
            </p>
            <div style="margin-top: 10px; font-size: 12px; color: #c4d7f2; display: flex; gap: 20px;">
                <span><strong>Institution:</strong> Dev Bhoomi Uttarakhand University (DBUU)</span>
                <span><strong>Framework:</strong> MediaPipe + Scikit-learn + OpenCV</span>
                <span><strong>Scope:</strong> Academic Early-Warning Prototype</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_state_badge(state: str, is_alert_active: bool):
    """
    Renders high-contrast driver state indicators.
    """
    if state == settings.STATE_ALERT:
        bg_color = "#28a745"
        text = "DRIVER STATE: ALERT"
        sub = "Normal driver alertness confirmed. Continuous passive monitoring active."
    elif state == settings.STATE_WARNING:
        bg_color = "#fd7e14"
        text = "DRIVER STATE: WARNING"
        sub = "Early fatigue signals observed. Temporal threshold not yet breached."
    elif state == settings.STATE_DROWSY:
        bg_color = "#dc3545"
        alarm_txt = " [AUDIBLE ALARM ACTIVE]" if is_alert_active else ""
        text = f"DRIVER STATE: DROWSY{alarm_txt}"
        sub = "SUSTAINED DROWSINESS CONFIRMED! Audio and visual alarms activated."
    elif state == settings.STATE_TRACKING_LOST:
        bg_color = "#6c757d"
        text = "DRIVER STATE: TRACKING LOST"
        sub = "Driver face not visible. Tracking failure flagged; fatigue score held."
    else:
        bg_color = "#343a40"
        text = f"DRIVER STATE: {state}"
        sub = "System initializing..."

    st.markdown(
        f"""
        <div style="background-color: {bg_color}; padding: 16px; border-radius: 8px; color: white; text-align: center; margin-bottom: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.15);">
            <h3 style="margin: 0; font-size: 22px; font-weight: 800; letter-spacing: 0.5px;">{text}</h3>
            <p style="margin: 4px 0 0 0; font-size: 13px;">{sub}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_metric_cards(features: dict, ml_pred: dict, warning_score: float, fps: float, latency_ms: float):
    """
    Renders structured real-time metric cards.
    """
    c1, c2, c3, c4 = st.columns(4)

    ear = features.get("ear", 0.0)
    mar = features.get("mar", 0.0)
    pitch = features.get("head_pitch", 0.0)
    roll = features.get("head_roll", 0.0)
    blinks = features.get("blink_rate", 0.0)
    closure_dur = features.get("eye_closure_duration", 0)

    with c1:
        st.metric(
            label="Eye Aspect Ratio (EAR)",
            value=f"{ear:.3f}",
            delta=f"Threshold: {settings.EAR_DROWSY_THRESHOLD}",
            delta_color="off"
        )
        st.metric(
            label="Closure Duration",
            value=f"{closure_dur} frames",
            delta=f"Microsleep >= {settings.PROLONGED_CLOSURE_MIN_FRAMES}",
            delta_color="inverse" if closure_dur >= settings.PROLONGED_CLOSURE_MIN_FRAMES else "off"
        )

    with c2:
        st.metric(
            label="Mouth Aspect Ratio (MAR)",
            value=f"{mar:.3f}",
            delta=f"Yawn >= {settings.MAR_YAWN_THRESHOLD}",
            delta_color="inverse" if mar >= settings.MAR_YAWN_THRESHOLD else "off"
        )
        st.metric(
            label="Blink Frequency",
            value=f"{blinks:.1f} / min",
            delta="Normal: 12-24/min",
            delta_color="off"
        )

    with c3:
        st.metric(
            label="Head Pitch (Nodding)",
            value=f"{pitch:.1f}°",
            delta=f"Down >= {abs(settings.DOWNWARD_PITCH_THRESHOLD)}°" if pitch < settings.DOWNWARD_PITCH_THRESHOLD else "Normal orientation",
            delta_color="inverse" if pitch < settings.DOWNWARD_PITCH_THRESHOLD else "off"
        )
        st.metric(
            label="Head Roll (Tilt)",
            value=f"{roll:.1f}°",
            delta="Lateral tilt angle",
            delta_color="off"
        )

    with c4:
        st.metric(
            label="Temporal Warning Score",
            value=f"{warning_score:.1f} / {settings.DROWSY_SCORE_THRESHOLD}",
            delta=f"Warning at {settings.WARNING_SCORE_THRESHOLD}",
            delta_color="inverse" if warning_score >= settings.WARNING_SCORE_THRESHOLD else "off"
        )
        conf = ml_pred.get("confidence", 0.0) * 100.0
        ml_state = ml_pred.get("state", "N/A")
        st.metric(
            label=f"ML Pred ({ml_state})",
            value=f"{conf:.1f}% conf",
            delta=f"FPS: {fps:.1f} | {latency_ms:.0f}ms",
            delta_color="off"
        )
