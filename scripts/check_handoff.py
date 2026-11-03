"""Dataset-free BE handoff checks; defaults must work without a pilot SVM."""
import json
from fastapi import FastAPI
from fastapi.testclient import TestClient
from api.main import app
from ai.paths import ROOT
from integration.backend_proxy import create_ai_router


def main():
    checks = []
    with TestClient(app) as client:
        health = client.get('/health').json()
        assert health['model_provider'] == 'onnx_alphabet', health
        checks.append('default provider is ONNX, not missing pilot SVM')
        caps = client.get('/api/v1/ai/capabilities').json()
        assert len(caps['static_targets']) == 24 and set(caps['dynamic_letter_targets']) == {'J', 'Z'}
        assert caps['dynamic_letter_modes']['Z'] == 'onnx_DZ_handshape_and_filtered_index_trajectory'
        checks.append('static/J/Z contract; ONNX and path gates mandatory')
        assert '/api/v1/ai/predict/image' not in client.get('/openapi.json').json()['paths']
        assert client.get('/demo/').status_code == 404
        checks.append('API-only package has no image or demo routes')
        for name, endpoint, expected in [
            ('onnx_example_request.json', 'landmarks', 'A'),
            ('jz_example_request.json', 'attempt', 'J'),
            ('z_example_request.json', 'attempt', 'Z'),
        ]:
            payload = json.loads((ROOT/'integration'/name).read_text())
            response = client.post('/api/v1/ai/predict/'+endpoint, json=payload)
            assert response.status_code == 200, response.text
            result = response.json()
            assert result['status'] == 'recognized' and result['predicted_label'] == expected, result
            assert result['error_type'] is None and not result['diagnosis_supported']
            if expected != 'A':
                assert result['model_confidence'] is None
            checks.append(f'{expected} frozen fixture through actual API')
            (ROOT/'integration'/name.replace('request', 'response')).write_text(json.dumps(result, indent=2))
        payload = json.loads((ROOT/'integration/onnx_example_request.json').read_text())
        for target, points, expected in [('A', None, 'no_hand'), ('HELLO', payload['landmarks'], 'unsupported_target')]:
            result = client.post('/api/v1/ai/predict/landmarks', json={**payload, 'target_label': target, 'landmarks': points}).json()
            assert result['status'] == expected and result['matches_target'] is None
        checks.append('no-hand and unsupported words are not learner errors')
        (ROOT/'integration/openapi.json').write_text(json.dumps(client.get('/openapi.json').json(), indent=2))
    backend = FastAPI()
    backend.include_router(create_ai_router())
    assert '/api/v1/ai/predict/attempt' in backend.openapi()['paths']
    checks.append('BE router mounts and exposes attempt endpoint')
    report = {'passed': len(checks), 'checks': checks, 'model_version': health['model_version'],
              'dataset_required': False, 'synthetic_motion_fixtures': True, 'webcam_accuracy_evaluated': False}
    (ROOT/'reports').mkdir(exist_ok=True)
    (ROOT/'reports/handoff_checks.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == '__main__':
    main()
