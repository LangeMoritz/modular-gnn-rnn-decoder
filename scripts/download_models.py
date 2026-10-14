"""Download the trained models of the paper from Zenodo into models/ and check their hashes.

  python scripts/download_models.py            # all models
  python scripts/download_models.py memory_d9  # selected ones
"""
import hashlib
import sys
import urllib.request
from pathlib import Path

RECORD = "https://zenodo.org/records/23335425/files"
MODELS = {  # name: sha256
    "google_d3_q10_7": "a6ddcf8fcc345501d92074dd2144a9274de71519eeccd185898241e215864c10",
    "google_d3_q2_7": "af6f288dce98b9ca8fa1558019adafbadb67ae209ae4a9898d439873a55f86f9",
    "google_d3_q4_5": "c4f5a35dd5d68753ba089a27dd90531323a581e91c53b027ffc720c4cdf40faf",
    "google_d3_q4_9": "775d1076d714690626c3eb551a9698b60c4d318220dd5fe9a359cec8aae32c8f",
    "google_d3_q6_11": "a8b2eedc5f80288ae162bdba417ad7e8ba02d42f049f3520bc19d84c037af977",
    "google_d3_q6_3": "4945ca67d2baf8d693eee72a4c7745132924f39870b5ed24185e019f6896a412",
    "google_d3_q6_7": "b84fdbb1a52238f59b85361cf23bbb12cd13039b1b35a28c170d6dbbec6713d9",
    "google_d3_q8_5": "4fdc180ad1ae5d4e9ff14aef5a8ff3be858b450b922af3746fea9fd9efa519a3",
    "google_d3_q8_9": "120839d2e4bde0b2cef7373e484b3a48429aaae53a63621a1e92a46a99662474",
    "google_d5_q4_7": "6547dd121aa289a0a84505fb12881a57ef0125c72f5d72968ae73e38148a4996",
    "google_d5_q6_5": "799a7aeef26c041dafb9e8cebd4078c47962262e108fa995b44be8d66fd55626",
    "google_d5_q6_9": "1fbe5b2d6cbbff880c1ff938212e42888f29cde1f2a5b5159efe21338d1f3358",
    "google_d5_q8_7": "b8c3f294e944cb080607c626c4d885a39ca67fe4f3da224dae3c87e3669cd53e",
    "google_d7_q6_7": "26f67f1c4ef353c2ce283c445f3baa7c1bcf1b1304740169561c8f7f45d3c80d",
    "google_d7_q6_7_wide": "cfae5a08350f55716659d11a3cf88df34b9e243f344e64d5b289570300bb0762",
    "google_si1000_d3": "c83ec6e1d77ce39011b4420f82b51194f551b72ff55ea2de9eea8f4f379825f2",
    "ibm_memory_d3_q36": "56d0eeece203545fb9f7c156972ff8733aac0a81aaf6f1a12971e64169d94a22",
    "ibm_memory_d3_q85": "807e14f6afbc5ffd045331099f9f93caf1078c6e03d0701f1f538b2ce3b2f17c",
    "ibm_memory_d5_q85": "c1242a695c401f4098dd671f0d90596056b97f0006b6772c7494c625b589ea04",
    "ibm_stability_d3": "16f9dceac9bb2f3b1ae5f0c0f512958a3a70add3c4d8aaa8cf08f4eb5f864910",
    "ibm_zz_merge": "7a29b3526f765c6f5960b9a00b1c74c2bab935971df866d206a0cbab79b25949",
    "memory_d3": "838a33db2619debcfde0cca84a20f72808923eb4b8485c2c5b4aa9de066acc9d",
    "memory_d5": "0d93704e988f93576cfe44e66fce1f4d9d633e3835878623816042c9cddc6400",
    "memory_d7": "38c9e8e04f2b0af9b0e5d6d0cadc26ec289d4387e6aa185ee59f337b3cdd9d58",
    "memory_d9": "4f925712e7e12e616d10891af3b48d05da9b52f9a2fb245a5accad53003c0fa1",
    "merge_d3": "04398625f289499911a5d64dafb9fa9d575eb4a527d78b0abc7aa3681782ce38",
    "stability_d3": "a87212dad6b552e83c1043e052ddcf6ad9c3d8b82bce83006315079fad41b46f",
}

out = Path(__file__).resolve().parent.parent / "models"
out.mkdir(exist_ok=True)
for name in sys.argv[1:] or MODELS:
    path = out / f"{name}.pt"
    if not path.exists():
        print(f"downloading {name}.pt")
        urllib.request.urlretrieve(f"{RECORD}/{name}.pt?download=1", path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != MODELS[name]:
        sys.exit(f"{path}: checksum mismatch, delete it and download again")
print(f"models in {out}")
