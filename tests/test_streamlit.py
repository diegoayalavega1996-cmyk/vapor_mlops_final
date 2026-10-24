from streamlit.testing.v1 import AppTest

from vapor_mlops.config import ROOT_DIR


def test_streamlit_opens_and_predicts():
    app = AppTest.from_file(str(ROOT_DIR / "app/streamlit_app.py"), default_timeout=60).run()
    assert not app.exception
    assert len(app.number_input) == 14
    app.button(key="predict_button").click().run()
    assert not app.exception
    assert any(metric.label == "Consumo predicho (t/h)" for metric in app.metric)
