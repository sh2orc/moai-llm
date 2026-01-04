#!/bin/bash

# =============================================================================
# Moai-LLM Tokenized Dataset Cache 다운로드 스크립트
# Hugging Face Hub에서 sh2orc/moai-llm-tokenized-dataset-cache 다운로드
# =============================================================================

set -e

# 스크립트가 위치한 디렉토리로 이동
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

REPO_ID="sh2orc/moai-llm-tokenized-dataset-cache"
REPO_TYPE="dataset"

echo "=============================================="
echo "Moai-LLM Tokenized Dataset Cache 다운로드"
echo "Repository: $REPO_ID"
echo "Target Directory: $SCRIPT_DIR"
echo "=============================================="

# 다운로드 방법 선택
download_with_huggingface_cli() {
    echo ""
    echo "[방법 1] huggingface-cli 사용..."
    
    # huggingface-cli 설치 확인
    if ! command -v huggingface-cli &> /dev/null; then
        echo "huggingface-cli가 설치되어 있지 않습니다."
        echo "설치 중: pip install huggingface_hub"
        pip install huggingface_hub
    fi
    
    # 데이터셋 다운로드
    huggingface-cli download "$REPO_ID" \
        --repo-type "$REPO_TYPE" \
        --local-dir "$SCRIPT_DIR/data" \
        --local-dir-use-symlinks False
    
    echo ""
    echo "다운로드 완료! 저장 위치: $SCRIPT_DIR/data"
}

download_with_git_lfs() {
    echo ""
    echo "[방법 2] git lfs 사용..."
    
    # git lfs 설치 확인
    if ! command -v git-lfs &> /dev/null; then
        echo "git-lfs가 설치되어 있지 않습니다."
        echo "macOS: brew install git-lfs"
        echo "Ubuntu: sudo apt-get install git-lfs"
        exit 1
    fi
    
    # git lfs 초기화
    git lfs install
    
    # 클론
    if [ -d "$SCRIPT_DIR/data" ]; then
        echo "data 디렉토리가 이미 존재합니다. 삭제 후 다시 클론합니다."
        rm -rf "$SCRIPT_DIR/data"
    fi
    
    git clone "https://huggingface.co/datasets/$REPO_ID" "$SCRIPT_DIR/data"
    
    echo ""
    echo "다운로드 완료! 저장 위치: $SCRIPT_DIR/data"
}

download_with_python() {
    echo ""
    echo "[방법 3] Python datasets 라이브러리 사용..."
    
    python3 << EOF
from huggingface_hub import snapshot_download
import os

save_path = os.path.join("$SCRIPT_DIR", "data")
os.makedirs(save_path, exist_ok=True)

print(f"다운로드 시작: $REPO_ID")
snapshot_download(
    repo_id="$REPO_ID",
    repo_type="$REPO_TYPE",
    local_dir=save_path,
    local_dir_use_symlinks=False
)
print(f"다운로드 완료! 저장 위치: {save_path}")
EOF
}

# 메인 실행
echo ""
echo "다운로드 방법을 선택하세요:"
echo "1) huggingface-cli (권장)"
echo "2) git lfs"
echo "3) Python (huggingface_hub)"
echo ""

# 인자가 있으면 해당 방법 사용, 없으면 기본값 1
METHOD="${1:-1}"

case "$METHOD" in
    1)
        download_with_huggingface_cli
        ;;
    2)
        download_with_git_lfs
        ;;
    3)
        download_with_python
        ;;
    *)
        echo "잘못된 옵션입니다. 1, 2, 3 중 하나를 선택하세요."
        exit 1
        ;;
esac

echo ""
echo "=============================================="
echo "완료!"
echo "=============================================="

