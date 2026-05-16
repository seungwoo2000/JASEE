import http.server
import socketserver
import os
import urllib.parse
import webbrowser
import sys

# 설정
PORT = 8000
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
IMAGE_DIRS = [
    os.path.join(ROOT_DIR, "new images_data/images/GOOD"),
    os.path.join(ROOT_DIR, "new images_data/images/BAD"),
    os.path.join(ROOT_DIR, "FOR_DA/FOR_DA/YOLO_full_body"),
    os.path.join(ROOT_DIR, "FOR_DA/FOR_DA/YOLO_ankle_visible")
]

class ViewerHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path
        query = urllib.parse.parse_qs(parsed_url.query)

        if path == "/get_image":
            filename = query.get("filename", [None])[0]
            if not filename:
                self.send_error(400, "Missing filename")
                return

            # 이미지 검색
            image_path = None
            for d in IMAGE_DIRS:
                trial = os.path.join(d, filename)
                if os.path.exists(trial):
                    image_path = trial
                    break
            
            if image_path:
                self.send_response(200)
                self.send_header("Content-type", "image/jpeg") # JPEG/PNG 지원
                self.end_headers()
                with open(image_path, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.send_error(404, f"Image not found: {filename}")
            return

        # 기본 동작 (파일 서빙)
        return super().do_GET()

    def translate_path(self, path):
        # viewer.html이 있는 디렉토리를 기준으로 서빙
        return super().translate_path(path)

if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))
    
    with socketserver.TCPServer(("", PORT), ViewerHandler) as httpd:
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f" Pose Landmark Viewer Server")
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f" 주소: http://localhost:{PORT}/viewer.html")
        print(f" 루트: {ROOT_DIR}")
        print(f" 중단하려면 Ctrl+C를 누르세요.")
        
        webbrowser.open(f"http://localhost:{PORT}/viewer.html")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n서버를 종료합니다.")
            sys.exit(0)
