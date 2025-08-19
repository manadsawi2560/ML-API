from flask import Flask,request
from flask_restful import Resource, Api
import pickle
import pandas as pd
from flask_cors import CORS
from flask import jsonify


app = Flask(__name__)
#
CORS(app)
# creating an API object
api = Api(app)

# --- load model once ---
with open('simple_linear_regression.pkl', 'rb') as f:
    model = pickle.load(f)

def _get_expected_feature_name(m):
    # พยายามอ่านชื่อฟีเจอร์จากตัวโมเดลหรือขั้นสุดท้ายของ Pipeline
    if hasattr(m, "feature_names_in_"):
        return list(m.feature_names_in_)
    if hasattr(m, "steps") and hasattr(m.steps[-1][1], "feature_names_in_"):
        return list(m.steps[-1][1].feature_names_in_)
    # ถ้าอ่านไม่ได้ ให้ fallback เป็นคีย์คำที่น่าจะใช้งาน
    return ["Marketing Budget"]

EXPECTED_FEATURES = _get_expected_feature_name(model)
# สำหรับเคสนี้ควรมีเพียง 1 ฟีเจอร์ (simple linear regression)
EXPECTED_FEATURE = EXPECTED_FEATURES[0] if len(EXPECTED_FEATURES) >= 1 else "Marketing Budget"

# prediction api call
class prediction(Resource):
    def get(self, budget):
        try:
            # ตรวจและแปลงเป็นตัวเลข
            val = float(budget)
        except (TypeError, ValueError):
            return jsonify({"error": "budget ต้องเป็นตัวเลข เช่น 12345"}), 400

        # สร้าง DataFrame โดยใช้ 'EXPECTED_FEATURE' ให้ตรงกับตอน fit
        df = pd.DataFrame({EXPECTED_FEATURE: [val]})

        try:
            y = model.predict(df)
            pred = float(y[0])
        except Exception as e:
            return jsonify({"error": f"prediction failed: {e}"}), 500

        return str(int(pred))

# data api
class getData(Resource):
    def get(self):
        try:
            df = pd.read_excel('data.xlsx') 
        except Exception as e:
            return jsonify({"error": f"read excel failed: {e}"}), 500

        # รีเนมเพื่อให้ front-end อ่านง่าย
        df = df.rename(
            {'Marketing Budget': 'budget', 'Actual Sales': 'sale'},
            axis=1
        )

        # ส่งออกเป็น JSON records
        return df.to_json(orient='records')
    
#
api.add_resource(getData, '/api')
api.add_resource(prediction, '/prediction/<int:budget>')

if __name__ == '__main__':
    app.run(debug=True)