# Implementation Plan: Qwen Image Layered AI Agent System

## 구현 계획서
**Version:** 1.0
**Date:** 2026-01-09
**Status:** Ready for Implementation

---

## 1. Implementation Overview

### 1.1 Summary

| Item | Count | Priority |
|------|-------|----------|
| Skills | 4개 | P0 |
| SubAgents | 4개 | P0 |
| Support Files | 3개 | P0 |

### 1.2 Implementation Order

```
Phase 1: Foundation
    └── CLAUDE.md 생성
    └── 디렉토리 구조 생성

Phase 2: Skills (자동 트리거)
    └── image-decompose
    └── layer-export
    └── quick-edit
    └── background-remove

Phase 3: SubAgents (작업 위임)
    └── batch_processor
    └── layer_editor
    └── composition_engine
    └── quality_checker

Phase 4: Integration & Testing
    └── 통합 테스트
    └── 문서 업데이트
```

---

## 2. Phase 1: Foundation

### Task 1.1: Directory Structure

```bash
# 실행할 명령어
mkdir -p .claude/agents
mkdir -p .claude/skills/image-decompose
mkdir -p .claude/skills/layer-export
mkdir -p .claude/skills/quick-edit
mkdir -p .claude/skills/background-remove
```

**생성될 구조:**
```
.claude/
├── CLAUDE.md
├── agents/
│   ├── batch_processor.md
│   ├── layer_editor.md
│   ├── composition_engine.md
│   └── quality_checker.md
└── skills/
    ├── image-decompose/
    │   └── SKILL.md
    ├── layer-export/
    │   └── SKILL.md
    ├── quick-edit/
    │   └── SKILL.md
    └── background-remove/
        └── SKILL.md
```

### Task 1.2: CLAUDE.md

**File:** `.claude/CLAUDE.md`

**내용:** 프로젝트 컨텍스트, 사용 가능한 Skills/SubAgents 목록, 주요 명령어

---

## 3. Phase 2: Skills Implementation

### Task 2.1: image-decompose Skill

**Priority:** P0 (Core Feature)

**File:** `.claude/skills/image-decompose/SKILL.md`

**Triggers:**
- "분해", "decompose", "레이어로 나눠"
- "이미지를 N개 레이어로"

**Implementation Steps:**
1. SKILL.md 파일 생성
2. 트리거 키워드 정의
3. 입력/출력 파라미터 정의
4. 실행 스크립트 연결

**Dependencies:**
- `src/fal_api/decompose.py` (기존)
- `src/fal_api/run_demo.py` (기존)

**Test Cases:**
- [ ] "이미지를 4개 레이어로 분해해줘" → 성공
- [ ] "test.png 분해" → 성공
- [ ] "레이어로 나눠줘" → 이미지 요청

---

### Task 2.2: layer-export Skill

**Priority:** P0 (Core Feature)

**File:** `.claude/skills/layer-export/SKILL.md`

**Triggers:**
- "내보내기", "export", "저장"
- "PPTX로", "PSD로", "ZIP으로"

**Implementation Steps:**
1. SKILL.md 파일 생성
2. 형식 선택 로직 정의
3. LayerExporter 연결

**Dependencies:**
- `src/fal_api/export.py` (기존)

**Test Cases:**
- [ ] "PPTX로 내보내줘" → PPTX 생성
- [ ] "모든 형식으로 저장" → PNG, PPTX, PSD, ZIP 생성

---

### Task 2.3: quick-edit Skill

**Priority:** P1 (Enhancement)

**File:** `.claude/skills/quick-edit/SKILL.md`

**Triggers:**
- "색상 변경", "크기 조절", "위치 이동"

**Implementation Steps:**
1. SKILL.md 파일 생성
2. PIL 기반 편집 로직 추가
3. 파라미터 파싱 로직

**Dependencies:**
- Pillow (기존)

**Test Cases:**
- [ ] "레이어 1 색상을 빨간색으로" → 색상 변경
- [ ] "레이어 2 크기 50%로" → 리사이즈

---

### Task 2.4: background-remove Skill

**Priority:** P1 (Enhancement)

**File:** `.claude/skills/background-remove/SKILL.md`

**Triggers:**
- "배경 제거", "누끼", "배경 분리"

**Implementation Steps:**
1. SKILL.md 파일 생성
2. 2레이어 분해 모드 연결

**Dependencies:**
- image-decompose skill (layers=2)

**Test Cases:**
- [ ] "배경 제거해줘" → 전경/배경 2레이어
- [ ] "누끼 따줘" → 전경/배경 2레이어

---

## 4. Phase 3: SubAgents Implementation

### Task 3.1: BatchProcessor SubAgent

**Priority:** P0 (Core Feature)

**File:** `.claude/agents/batch_processor.md`

**Responsibility:**
- 여러 이미지 일괄 처리
- 진행 상황 추적
- 실패 항목 재시도

**Implementation Steps:**
1. Agent 정의 파일 생성
2. 워크플로우 정의
3. 에러 핸들링 로직

**When Triggered:**
- 2개 이상 이미지 처리 요청 시
- 폴더/디렉토리 처리 요청 시

**Test Cases:**
- [ ] "폴더의 모든 이미지 분해" → 배치 처리
- [ ] 10개 이미지 일괄 처리 → 결과 리포트

---

### Task 3.2: LayerEditor SubAgent

**Priority:** P1 (Enhancement)

**File:** `.claude/agents/layer_editor.md`

**Responsibility:**
- 복잡한 멀티스텝 편집
- Qwen-Image-Edit 연동
- 편집 히스토리 관리

**Implementation Steps:**
1. Agent 정의 파일 생성
2. 지원 작업 목록 정의
3. 상태 관리 로직

**When Triggered:**
- 여러 편집 작업 연속 요청 시
- 복잡한 편집 파이프라인 시

**Test Cases:**
- [ ] "레이어 1 색상 변경 후 크기 조절" → 순차 편집
- [ ] 편집 취소 → Undo 동작

---

### Task 3.3: CompositionEngine SubAgent

**Priority:** P1 (Enhancement)

**File:** `.claude/agents/composition_engine.md`

**Responsibility:**
- 레이어 합성
- 무드보드/콜라주 생성
- 블렌딩 모드 처리

**Implementation Steps:**
1. Agent 정의 파일 생성
2. 합성 모드 정의
3. 템플릿 시스템

**When Triggered:**
- 레이어 병합 요청 시
- 무드보드/콜라주 생성 시

**Test Cases:**
- [ ] "레이어 합치기" → Alpha Composite
- [ ] "무드보드 만들어줘" → 그리드 레이아웃

---

### Task 3.4: QualityChecker SubAgent

**Priority:** P2 (Nice to Have)

**File:** `.claude/agents/quality_checker.md`

**Responsibility:**
- 분해 결과 품질 검증
- 자동 재처리 권장
- 품질 리포트 생성

**Implementation Steps:**
1. Agent 정의 파일 생성
2. 검증 기준 정의
3. 점수 시스템

**When Triggered:**
- 분해 완료 후 자동
- 사용자가 품질 검증 요청 시

**Test Cases:**
- [ ] 정상 분해 → Score 90+
- [ ] 빈 레이어 포함 → Warning 표시

---

## 5. Phase 4: Integration & Testing

### Task 4.1: Integration Testing

**테스트 시나리오:**

| # | Scenario | Expected Result |
|---|----------|-----------------|
| 1 | "이미지 분해해줘" | image-decompose skill 실행 |
| 2 | "폴더 전체 분해" | BatchProcessor 위임 |
| 3 | "분해 후 PPTX로 저장" | 연속 skill 실행 |
| 4 | "배경 제거하고 색상 변경" | skill → SubAgent |

### Task 4.2: Documentation Update

**업데이트할 문서:**
- README.md - Skills/SubAgents 사용법 추가
- docs/PRD.md - 최종 검토
- docs/LLD.md - 최종 검토

---

## 6. Implementation Checklist

### Phase 1: Foundation
- [ ] 디렉토리 구조 생성
- [ ] CLAUDE.md 생성

### Phase 2: Skills
- [ ] image-decompose SKILL.md
- [ ] layer-export SKILL.md
- [ ] quick-edit SKILL.md
- [ ] background-remove SKILL.md

### Phase 3: SubAgents
- [ ] batch_processor.md
- [ ] layer_editor.md
- [ ] composition_engine.md
- [ ] quality_checker.md

### Phase 4: Integration
- [ ] 통합 테스트
- [ ] 문서 업데이트
- [ ] Git commit & push

---

## 7. File Creation Order

| Order | File | Type |
|-------|------|------|
| 1 | `.claude/CLAUDE.md` | Config |
| 2 | `.claude/skills/image-decompose/SKILL.md` | Skill |
| 3 | `.claude/skills/layer-export/SKILL.md` | Skill |
| 4 | `.claude/skills/quick-edit/SKILL.md` | Skill |
| 5 | `.claude/skills/background-remove/SKILL.md` | Skill |
| 6 | `.claude/agents/batch_processor.md` | SubAgent |
| 7 | `.claude/agents/layer_editor.md` | SubAgent |
| 8 | `.claude/agents/composition_engine.md` | SubAgent |
| 9 | `.claude/agents/quality_checker.md` | SubAgent |

---

## 8. Dependencies Check

### Required (이미 구현됨)
- [x] `src/fal_api/decompose.py`
- [x] `src/fal_api/export.py`
- [x] `src/fal_api/run_demo.py`
- [x] `src/fal_api/app_test.py`
- [x] `.env.example`
- [x] `requirements-fal.txt`

### To Be Created
- [ ] `.claude/CLAUDE.md`
- [ ] `.claude/skills/**/SKILL.md` (4개)
- [ ] `.claude/agents/*.md` (4개)

---

## 9. Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Skill 트리거 충돌 | 키워드 우선순위 정의 |
| SubAgent 무한 루프 | 재귀 호출 제한 |
| API 비용 초과 | 사용량 모니터링 |

---

## 10. Success Criteria

| Criteria | Target |
|----------|--------|
| 모든 Skills 동작 | 4/4 |
| 모든 SubAgents 동작 | 4/4 |
| 통합 테스트 통과 | 100% |
| 문서화 완료 | 100% |

---

## Next Step

**준비가 되면 다음 명령어로 구현을 시작합니다:**

```
"Phase 1부터 순차적으로 구현해주세요"
```

또는 특정 단계만:

```
"Phase 2: Skills만 먼저 구현해주세요"
```
