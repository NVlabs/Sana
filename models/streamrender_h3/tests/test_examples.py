import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('prepare_example', ROOT / 'scripts/prepare_example.py')
examples = importlib.util.module_from_spec(spec)
spec.loader.exec_module(examples)


class ExampleTests(unittest.TestCase):
    def test_catalog_and_default(self):
        rows = examples.requests()
        self.assertEqual(len(rows), 60)
        self.assertEqual(len({r['name'] for r in rows}), 60)
        self.assertEqual(len({r['reference_video_path'] for r in rows}), 30)
        manifest = json.loads((ROOT / 'configs/assets.example.json').read_text())
        default = next(r for r in rows if r['name'] == examples.DEFAULT)
        self.assertEqual(manifest['assets']['prompt'], default['prompt'])
        for row in rows:
            for field in ('picture', 'reference_video_path'):
                path = Path(row[field])
                self.assertFalse(path.is_absolute())
                self.assertNotIn('..', path.parts)

    def test_manifest_relocation_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'weights/assets.json'
            source.parent.mkdir()
            source.write_text(json.dumps({'assets': {'backbone': 'backbone/model.safetensors',
                                                    'prompt': 'old', 'reference_image': 'old.png'}}))
            row = next(r for r in examples.requests() if r['name'] == examples.DEFAULT)
            for key in ('picture', 'reference_video_path'):
                path = root / row[key]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            output = root / 'configs/test.local.json'
            examples.prepare(source, examples.DEFAULT, root, output)
            assets = json.loads(output.read_text())['assets']
            self.assertEqual(assets['backbone'], str(source.parent / 'backbone/model.safetensors'))
            self.assertEqual(assets['prompt'], row['prompt'])
            self.assertEqual(assets['reference_image'], str(root / row['picture']))
            with self.assertRaises(FileExistsError):
                examples.prepare(source, examples.DEFAULT, root, output)
