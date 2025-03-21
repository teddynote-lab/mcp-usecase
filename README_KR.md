# Quick-start Auto MCP : All in one Claude Desktop and Cursor

[English](README.md) | [한국어](README_KR.md)

## 소개

**Quick-start Auto MCP**는 Anthropic의 Model Context Protocol(MCP)을 Claude Desktop과 Cursor에 쉽고 빠르게 등록하여 사용할 수 있도록 도와주는 도구입니다.

**주요 장점:**
1. **빠른 설정**: 간단한 도구 실행 및 JSON 파일 복사/붙여넣기만으로 Claude Desktop과 Cursor에 MCP 기능을 바로 추가할 수 있습니다.
2. **다양한 도구 제공**: 유용한 MCP 도구들을 지속적으로 업데이트합니다. Star와 Follow를 통해 나만의 도구를 꾸준히 업데이트 해보세요. :)

## 목차

- [특징](#-특징)
- [프로젝트 구조](#-프로젝트-구조)
- [요구사항](#-요구사항)
- [설치](#-설치)
- [설정](#-설정)
- [사용법](#-사용법)
- [문제해결](#-문제해결)
- [라이센스](#-라이센스)
- [기여하기](#-기여하기)
- [문의하기](#-문의하기)
- [저자](#-저자)

## 특징

- **RAG(Retrieval Augmented Generation)** - PDF 문서를 대상으로 키워드, 시맨틱, 하이브리드 검색 기능
- **Dify External Knowledge API** - Dify의 외부 지식 API를 통한 문서 검색 기능
- **Dify Workflow** - Dify Workflow 실행 및 결과 검색 기능
- **Web Search** - Exa API를 활용한 실시간 웹 검색 기능
- **자동 JSON 생성** - Claude Desktop과 Cursor에 필요한 MCP JSON 파일 자동 생성

## 프로젝트 구조

```
.
├── README.md                 
├── data                      # PDF 데이터 파일 
├── docs                      # 문서 폴더
│   ├── case1.md           # RAG 예제 설명
│   ├── case2.md           # Dify External Knowledge API 예제 설명
│   ├── case3.md           # Dify Workflow 예제 설명
│   ├── case4.md           # Web Search 예제 설명
│   └── installation.md    # 설치 가이드
├── case1                     # RAG 예제
├── case2                     # Dify External Knowledge API 예제
├── case3                     # Dify Workflow 예제
├── case4                     # Web Search 예제
├── pyproject.toml            # 프로젝트 설정
└── requirements.txt          # 필요 패키지 목록
```

## 요구사항

- Python >= 3.11
- Claude Desktop 또는 Cursor
- uv (권장) 또는 pip

## 설치

### 1. 저장소 복제

```bash
git clone https://github.com/teddynote-lab/mcp.git
cd mcp
```

### 2. 가상 환경 설정 

#### uv 사용 (권장)
```bash
uv venv
uv pip install -r requirements.txt
```

#### pip 사용
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. PDF 준비

RAG에 필요한 PDF 파일을 `./data`에 넣어주세요.

본 프로젝트에서는 [소프트웨어정책연구소(SPRi)의 25년 3월호](https://spri.kr/posts/view/23827?code=AI-Brief&s_year=&data_page=1) `인공지능 산업의 최신 동향`을 사용했습니다.

## 환경 변수 설정

각 예제 디렉토리에는 필요한 환경 변수를 설정할 `.env` 파일이 필요합니다. `.env.example`에 필요한 환경 변수를 설정하고 파일명을 `.env`로 변경해주세요.

### 예제별 필요한 환경 변수 설정 사이트
- https://platform.openai.com/api-keys
- https://dify.ai/
- https://app.tavily.com/home


## 사용법

### 1. JSON 파일 생성

각 예제 디렉토리에서 다음 명령을 실행하여 필요한 JSON 파일을 생성합니다:

```bash
# 가상 환경 활성화

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

# 예제 디렉토리로 이동
cd example1

# JSON 파일 생성
python auto_mcp_json.py
```

### 2. Claude Desktop/Cursor에 MCP 등록

1. Claude Desktop 또는 Cursor 실행
2. MCP 설정 메뉴 열기
3. 생성된 JSON 내용을 복사하여 붙여넣기
4. 저장 및 `재시작`

> **참고**: Claude Desktop 또는 Cursor를 실행하면 MCP 서버가 자동으로 함께 실행되며, 소프트웨어를 종료하면 MCP 서버도 함께 종료됩니다.

## Troubleshooting

일반적인 문제 및 해결 방법:

- **MCP 서버 연결 실패**: 서비스가 올바르게 실행 중인지, 포트가 충돌하지 않는지 확인하세요. 특히, case2를 적용할 때는 `dify_ek_server.py`를 같이 실행시켜주셔야 합니다.
- **API 키 오류**: 환경 변수가 올바르게 설정되었는지 확인하세요.
- **가상 환경 문제**: Python 버전이 3.11 이상인지 확인하세요.

## 라이센스

[MIT 라이센스](LICENSE.md)

## 기여하기

기여는 언제나 환영합니다! 이슈, 버그 또는 기능 추가에 대한 의견을 남겨주세요. :) 

## 문의하기

질문이나 도움이 필요하시면 이슈를 등록하거나 다음 연락처로 문의해 주세요:
dev@brain-crew.com

## 저자

[임한택](https://github.com/LHANTAEK)