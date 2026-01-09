# Implementation Plan: Qwen Image Layered AI Agent System

**Status**: In Progress
**Started**: 2026-01-09
**Last Updated**: 2026-01-09
**Estimated Completion**: 2026-01-16

---

**CRITICAL INSTRUCTIONS**: After completing each phase:
1. Check off completed task checkboxes
2. Run all quality gate validation commands
3. Verify ALL quality gate items pass
4. Update "Last Updated" date above
5. Document learnings in Notes section
6. Only then proceed to next phase

**DO NOT skip quality gates or proceed with failing checks**

---

## Overview

### Feature Description
Qwen Image Layered 프로젝트에 Claude Code Skills와 SubAgents 시스템을 구축합니다.
AGENTS.md 계층 구조로 규칙을 관리하고, TDD 방식으로 각 컴포넌트를 구현합니다.

### Success Criteria
- [ ] 4개 Skills 구현 및 자동 트리거 동작 확인
- [ ] 4개 SubAgents 구현 및 작업 위임 동작 확인
- [ ] AGENTS.md 계층 구조 완성
- [ ] 모든 Quality Gates 통과
- [ ] 통합 테스트 100% 통과

### User Impact
- 자연어 명령으로 이미지 레이어 분해 가능
- 복잡한 배치 작업 자동 위임
- 일관된 품질의 결과물 보장

---

## Architecture Decisions

| Decision | Rationale | Trade-offs |
|----------|-----------|------------|
| Skills 4개로 제한 | 핵심 기능에 집중, 복잡도 관리 | 확장성 제한 |
| SubAgents는 복잡한 작업만 | 단순 작업은 Skills로 처리 | 경계 판단 필요 |
| AGENTS.md 계층 구조 | 폴더별 상세 규칙 관리 | 파일 수 증가 |
| TDD 적용 | 품질 보장, 회귀 방지 | 초기 시간 투자 |

---

## Dependencies

### Required Before Starting
- [x] Fal AI 연동 코드 (`src/fal_api/`)
- [x] 테스트 UI (`src/fal_api/app_test.py`)
- [x] PRD, LLD 문서
- [ ] `.env` 파일에 FAL_KEY 설정

### External Dependencies
- fal-client >= 0.5.0
- python-dotenv >= 1.0.0
- Pillow >= 10.0.0
- pytest >= 7.0.0 (테스트용)

---

## Test Strategy

### Testing Approach
TDD Principle: 테스트를 먼저 작성하고, 구현으로 테스트 통과

### Test Structure
```
tests/
├── unit/
│   ├── test_skills.py
│   └── test_subagents.py
├── integration/
│   └── test_agent_system.py
└── e2e/
    └── test_full_workflow.py
```

### Coverage Requirements by Phase
- Phase 1 (Foundation): AGENTS.md 검증 스크립트
- Phase 2 (Skills): Skill 트리거 테스트 80%+
- Phase 3 (SubAgents): SubAgent 위임 테스트 80%+
- Phase 4 (Integration): E2E 테스트 1+ critical path

---

## File Creation Order

### AGENTS.md System (Phase 1)
```
1. ./AGENTS.md                           # Root - Central Control
2. ./src/AGENTS.md                       # Source Code Rules
3. ./src/fal_api/AGENTS.md              # Fal API Rules
4. ./.claude/AGENTS.md                   # Agent System Rules
```

### Skills (Phase 2)
```
5. ./.claude/skills/image-decompose/SKILL.md
6. ./.claude/skills/layer-export/SKILL.md
7. ./.claude/skills/quick-edit/SKILL.md
8. ./.claude/skills/background-remove/SKILL.md
```

### SubAgents (Phase 3)
```
9.  ./.claude/agents/batch_processor.md
10. ./.claude/agents/layer_editor.md
11. ./.claude/agents/composition_engine.md
12. ./.claude/agents/quality_checker.md
```

### Integration (Phase 4)
```
13. ./.claude/CLAUDE.md                  # Project Context (Update)
14. ./README.md                          # Documentation (Update)
```

---

## Implementation Phases

### Phase 1: AGENTS.md Foundation
**Goal**: 계층적 규칙 시스템 구축
**Estimated Time**: 2 hours
**Status**: Pending

#### Tasks

**RED: Write Validation Tests First**
- [ ] **Test 1.1**: AGENTS.md 파일 존재 검증 스크립트
  - File: `tests/test_agents_structure.py`
  - Expected: FAIL (파일 없음)
  - Details:
    - Root AGENTS.md 존재 확인
    - 필수 섹션 존재 확인
    - Context Map 유효성 검증

**GREEN: Implement AGENTS.md Files**
- [ ] **Task 1.2**: Root AGENTS.md 생성
  - File: `./AGENTS.md`
  - Content:
    - Project Context and Operations
    - Golden Rules (Immutable, Do's, Don'ts)
    - Standards and References
    - Context Map (Action-Based Routing)

- [ ] **Task 1.3**: Source AGENTS.md 생성
  - File: `./src/AGENTS.md`
  - Content:
    - Module Context
    - Tech Stack and Constraints
    - Implementation Patterns

- [ ] **Task 1.4**: Fal API AGENTS.md 생성
  - File: `./src/fal_api/AGENTS.md`
  - Content:
    - API 사용 규칙
    - 에러 핸들링 패턴
    - 비용 관리 지침

- [ ] **Task 1.5**: Claude Config AGENTS.md 생성
  - File: `./.claude/AGENTS.md`
  - Content:
    - Skills 작성 규칙
    - SubAgents 작성 규칙
    - 네이밍 컨벤션

**REFACTOR: Review and Optimize**
- [ ] **Task 1.6**: 500줄 제한 확인
- [ ] **Task 1.7**: 이모지 제거 확인
- [ ] **Task 1.8**: Context Map 링크 유효성 확인

#### Quality Gate

**STOP: Do NOT proceed to Phase 2 until ALL checks pass**

**Structure Compliance**:
- [ ] Root AGENTS.md 존재
- [ ] 모든 하위 AGENTS.md 존재
- [ ] 모든 파일 500줄 미만
- [ ] 이모지 사용 없음

**Content Compliance**:
- [ ] Golden Rules 섹션 존재
- [ ] Context Map 섹션 존재
- [ ] 모든 링크 유효

**Validation Commands**:
```bash
# 파일 존재 확인
ls -la ./AGENTS.md ./src/AGENTS.md ./src/fal_api/AGENTS.md ./.claude/AGENTS.md

# 줄 수 확인 (500줄 미만)
wc -l ./AGENTS.md ./src/AGENTS.md ./src/fal_api/AGENTS.md ./.claude/AGENTS.md

# 이모지 검색 (결과 없어야 함)
grep -P "[\x{1F300}-\x{1F9FF}]" ./AGENTS.md || echo "No emojis found"

# 테스트 실행
python -m pytest tests/test_agents_structure.py -v
```

**Manual Test Checklist**:
- [ ] AGENTS.md 파일 열어서 구조 확인
- [ ] Context Map 링크 클릭하여 이동 확인
- [ ] Golden Rules 내용 검토

---

### Phase 2: Skills Implementation
**Goal**: 4개 Skills 구현 및 자동 트리거 동작
**Estimated Time**: 4 hours
**Status**: Pending

#### Tasks

**RED: Write Failing Tests First**
- [ ] **Test 2.1**: Skills 트리거 테스트
  - File: `tests/unit/test_skills.py`
  - Expected: FAIL (Skills 없음)
  - Details:
    - image-decompose 트리거 키워드 테스트
    - layer-export 트리거 키워드 테스트
    - quick-edit 트리거 키워드 테스트
    - background-remove 트리거 키워드 테스트

**GREEN: Implement Skills**
- [ ] **Task 2.2**: image-decompose Skill
  - File: `.claude/skills/image-decompose/SKILL.md`
  - Triggers: 분해, decompose, 레이어로 나눠
  - Action: `python src/fal_api/run_demo.py --image {path} --layers {n}`

- [ ] **Task 2.3**: layer-export Skill
  - File: `.claude/skills/layer-export/SKILL.md`
  - Triggers: 내보내기, export, PPTX, PSD
  - Action: LayerExporter 호출

- [ ] **Task 2.4**: quick-edit Skill
  - File: `.claude/skills/quick-edit/SKILL.md`
  - Triggers: 색상 변경, 크기 조절
  - Action: PIL 기반 편집

- [ ] **Task 2.5**: background-remove Skill
  - File: `.claude/skills/background-remove/SKILL.md`
  - Triggers: 배경 제거, 누끼
  - Action: 2레이어 분해

**REFACTOR: Optimize Skills**
- [ ] **Task 2.6**: 트리거 키워드 중복 제거
- [ ] **Task 2.7**: 공통 패턴 추출
- [ ] **Task 2.8**: 에러 메시지 표준화

#### Quality Gate

**STOP: Do NOT proceed to Phase 3 until ALL checks pass**

**Skills Structure**:
- [ ] 4개 SKILL.md 파일 존재
- [ ] 각 Skill에 Triggers 섹션 존재
- [ ] 각 Skill에 Inputs/Outputs 정의

**Functionality**:
- [ ] image-decompose 트리거 동작
- [ ] layer-export 트리거 동작
- [ ] quick-edit 트리거 동작
- [ ] background-remove 트리거 동작

**Validation Commands**:
```bash
# Skills 파일 확인
ls -la .claude/skills/*/SKILL.md

# 테스트 실행
python -m pytest tests/unit/test_skills.py -v

# Skill 구조 검증
grep -l "triggers:" .claude/skills/*/SKILL.md
```

**Manual Test Checklist**:
- [ ] "이미지 분해해줘" 입력 시 image-decompose 활성화
- [ ] "PPTX로 내보내줘" 입력 시 layer-export 활성화
- [ ] "배경 제거해줘" 입력 시 background-remove 활성화

---

### Phase 3: SubAgents Implementation
**Goal**: 4개 SubAgents 구현 및 작업 위임 동작
**Estimated Time**: 4 hours
**Status**: Pending

#### Tasks

**RED: Write Failing Tests First**
- [ ] **Test 3.1**: SubAgents 구조 테스트
  - File: `tests/unit/test_subagents.py`
  - Expected: FAIL (SubAgents 없음)
  - Details:
    - batch_processor 정의 테스트
    - layer_editor 정의 테스트
    - composition_engine 정의 테스트
    - quality_checker 정의 테스트

**GREEN: Implement SubAgents**
- [ ] **Task 3.2**: BatchProcessor SubAgent
  - File: `.claude/agents/batch_processor.md`
  - Role: 여러 이미지 일괄 처리
  - Tools: execute, read, write, glob

- [ ] **Task 3.3**: LayerEditor SubAgent
  - File: `.claude/agents/layer_editor.md`
  - Role: 복잡한 멀티스텝 편집
  - Tools: execute, read, write

- [ ] **Task 3.4**: CompositionEngine SubAgent
  - File: `.claude/agents/composition_engine.md`
  - Role: 레이어 합성, 무드보드
  - Tools: execute, read, write

- [ ] **Task 3.5**: QualityChecker SubAgent
  - File: `.claude/agents/quality_checker.md`
  - Role: 품질 검증, 재처리 권장
  - Tools: execute, read

**REFACTOR: Optimize SubAgents**
- [ ] **Task 3.6**: 공통 워크플로우 패턴 추출
- [ ] **Task 3.7**: 에러 핸들링 표준화
- [ ] **Task 3.8**: 출력 포맷 통일

#### Quality Gate

**STOP: Do NOT proceed to Phase 4 until ALL checks pass**

**SubAgents Structure**:
- [ ] 4개 .md 파일 존재
- [ ] 각 SubAgent에 Role 정의
- [ ] 각 SubAgent에 Tools 정의
- [ ] 각 SubAgent에 Workflow 정의

**Functionality**:
- [ ] BatchProcessor 위임 테스트
- [ ] LayerEditor 위임 테스트
- [ ] CompositionEngine 위임 테스트
- [ ] QualityChecker 위임 테스트

**Validation Commands**:
```bash
# SubAgents 파일 확인
ls -la .claude/agents/*.md

# 테스트 실행
python -m pytest tests/unit/test_subagents.py -v

# 필수 섹션 확인
grep -l "Role" .claude/agents/*.md
grep -l "Tools" .claude/agents/*.md
```

**Manual Test Checklist**:
- [ ] "폴더의 모든 이미지 분해" → BatchProcessor 위임
- [ ] "레이어 색상 변경 후 크기 조절" → LayerEditor 위임
- [ ] "무드보드 만들어줘" → CompositionEngine 위임

---

### Phase 4: Integration and Testing
**Goal**: 전체 시스템 통합 및 검증
**Estimated Time**: 3 hours
**Status**: Pending

#### Tasks

**RED: Write Integration Tests**
- [ ] **Test 4.1**: E2E 워크플로우 테스트
  - File: `tests/e2e/test_full_workflow.py`
  - Expected: FAIL (통합 미완료)
  - Details:
    - Skill → API 호출 → 결과
    - SubAgent 위임 → 처리 → 보고

**GREEN: Complete Integration**
- [ ] **Task 4.2**: CLAUDE.md 업데이트
  - File: `.claude/CLAUDE.md`
  - Content: Skills/SubAgents 목록 추가

- [ ] **Task 4.3**: README.md 업데이트
  - File: `README.md`
  - Content: 에이전트 시스템 사용법 추가

- [ ] **Task 4.4**: 통합 테스트 실행
  - 모든 테스트 통과 확인

**REFACTOR: Final Polish**
- [ ] **Task 4.5**: 문서 일관성 검토
- [ ] **Task 4.6**: 불필요한 파일 정리
- [ ] **Task 4.7**: Git 커밋 및 푸시

#### Quality Gate

**FINAL: All checks must pass before completion**

**Integration**:
- [ ] Skills ↔ fal_api 연동 동작
- [ ] SubAgents ↔ Skills 연동 동작
- [ ] AGENTS.md 라우팅 정확

**Documentation**:
- [ ] CLAUDE.md 최신화
- [ ] README.md 최신화
- [ ] 모든 AGENTS.md 최신화

**Testing**:
- [ ] Unit tests 100% pass
- [ ] Integration tests 100% pass
- [ ] E2E tests 100% pass

**Validation Commands**:
```bash
# 전체 테스트 실행
python -m pytest tests/ -v --cov=src

# 커버리지 확인
python -m pytest tests/ --cov=src --cov-report=html

# Git 상태 확인
git status
```

**Manual Test Checklist**:
- [ ] 새 세션에서 "이미지 분해" 테스트
- [ ] 새 세션에서 "배치 처리" 테스트
- [ ] 새 세션에서 "무드보드 생성" 테스트

---

## Risk Assessment

| Risk | Probability | Impact | Mitigation Strategy |
|------|-------------|--------|---------------------|
| Skill 트리거 충돌 | Medium | Medium | 키워드 우선순위 정의, 테스트 |
| SubAgent 무한 루프 | Low | High | 재귀 호출 제한, 타임아웃 |
| AGENTS.md 동기화 실패 | Medium | Low | 자동 검증 스크립트 |
| Fal AI 비용 초과 | Low | Medium | 사용량 모니터링, 제한 설정 |

---

## Rollback Strategy

### If Phase 1 Fails
**Steps to revert**:
- `git checkout -- ./AGENTS.md ./src/AGENTS.md ./src/fal_api/AGENTS.md ./.claude/AGENTS.md`
- Remove any created directories

### If Phase 2 Fails
**Steps to revert**:
- `git checkout -- .claude/skills/`
- Restore to Phase 1 complete state

### If Phase 3 Fails
**Steps to revert**:
- `git checkout -- .claude/agents/`
- Restore to Phase 2 complete state

### If Phase 4 Fails
**Steps to revert**:
- `git checkout -- .claude/CLAUDE.md README.md`
- Restore to Phase 3 complete state

---

## Progress Tracking

### Completion Status
- **Phase 1 (AGENTS.md)**: 0%
- **Phase 2 (Skills)**: 0%
- **Phase 3 (SubAgents)**: 0%
- **Phase 4 (Integration)**: 0%

**Overall Progress**: 0% complete

### Time Tracking
| Phase | Estimated | Actual | Variance |
|-------|-----------|--------|----------|
| Phase 1 | 2 hours | - | - |
| Phase 2 | 4 hours | - | - |
| Phase 3 | 4 hours | - | - |
| Phase 4 | 3 hours | - | - |
| **Total** | 13 hours | - | - |

---

## Notes and Learnings

### Implementation Notes
- (To be filled during implementation)

### Blockers Encountered
- (To be filled during implementation)

### Improvements for Future Plans
- (To be filled after completion)

---

## References

### Documentation
- [PRD](../PRD.md)
- [LLD](../LLD.md)
- [Original PLAN](../PLAN.md)

### External
- [Claude Agent SDK](https://docs.anthropic.com/agent-sdk)
- [Fal AI API](https://fal.ai/models/fal-ai/qwen-image-layered)

---

## Final Checklist

**Before marking plan as COMPLETE**:
- [ ] All 4 phases completed with quality gates passed
- [ ] All 4 Skills functional
- [ ] All 4 SubAgents functional
- [ ] All AGENTS.md files created and valid
- [ ] Full integration testing performed
- [ ] Documentation updated
- [ ] Git commit and push completed
- [ ] Plan document archived

---

**Plan Status**: In Progress
**Next Action**: Phase 1 - AGENTS.md Foundation 구현
**Blocked By**: None
