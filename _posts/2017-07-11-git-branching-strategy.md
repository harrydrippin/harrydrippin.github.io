---
layout: post
title: "효율적인 협업을 위한 Git Branching 전략"
tags:
    - Engineering
---

회사에 다니기 시작한 이후 체계적인 Task 및 Sprint 관리와 함께 Git를 이용한 소프트웨어 개발을 시작했다.
평소에 경험해보지 못했던 애자일 방법론을 통한 소프트웨어 개발을 체험해보고 나서 팀 내에서 공유되는 코드들에 대한 체계적인 관리가 필요하다는 것을 느꼈다.

회사에서는 기존에 한 프로젝트를 완전히 한 명의 담당자가 처리했었기 때문에 특별한 Branch Model을 사용할 필요가 없었다.
그러나 최근 인원이 늘어나고 프로젝트별로 담당자가 몇 명씩 지정되면서 문제가 발생하기 시작했다. 각 Task별로 담당자가 나뉘어져 있고 같은 코드 안에서 여러 Task가 생기다 보니 많은 사람들이 하나의 Branch에 직접 Commit을 하며 매우 많은 Conflict가 일어났다.

갖가지 Conflict들이 작업을 지연시키는 것을 보고 적절한 협업 방법의 필요성이 생겨 자체적으로 Git Branching 전략을 만들었다.
현재는 우리 회사의 표준 Git 협업 전략으로써 사용되고 있고, 그 내용을 정리해보았다.

이 내용은 Miro Community의 [Git Branching Model][miro-community]과 Rumblefish 님이 정리한 [GitHub 기반 브랜치 명명 규칙][blog]을 기반으로 하여 회사에서 사용하는 도구들과 시스템에 맞게 재구성한 것이다.

## 상황

우리 회사는 [Atlassian JIRA][jira]를 사용하여 Task를 관리하며 애자일 방법론을 충실히 따라서 소프트웨어 개발 프로세스를 운영하고 있다.
시스템에서 모든 Task들과 Subtask들은 모두 Issue라는 이름으로 관리되며 버그 수정과 여타 다른 마일스톤들 역시 Issue로 통합되어 관리된다.

Issue는 **`[프로젝트 식별자]-XXX`**와 같은 형식으로 생성되며, XXX 자리에 생성 순서대로 숫자가 부여된다. 예를 들어 프로젝트 식별자가 POR이면, **`POR-237`**과 같은 형식이다.
이 글에서 말하는 Issue number는 여기서의 **237**을 말한다.

만약 JIRA와 함께 [Bitbucket][bitbucket]을 사용하고 있고 연동이 되어있다면 JIRA 애자일 보드에서 Issue를 선택하면 표시되는 사이드바의 **개발 > 분기점 만들기**를 통해서 Branch를 생성할 수 있다. 하지만 이 글에서 말하는 전략은 Bitbucket이 자동으로 생성해주는 Branch 이름 규칙과는 약간 다르다. 그러므로 이러한 형식으로 Branch를 만들고자 하는 경우에는 Branch의 이름 선정에 주의하여야 한다.

Github에서도 역시 Issue에 Label을 붙여서 Task들을 관리할 수 있다. Github는 **#1234**와 같은 형식으로 Issue Number를 관리하는데, Github 기반의 업무 환경에서는 이 숫자를 Issue Number로 한다.

구조 설명에 앞서, 전체적인 구조를 두 부분으로 나누어 생각하기로 하자. 첫 번째는 **소스 중앙 저장소**이며, 두 번째는 **개발자의 PC**이다.

### 소스 중앙 저장소

소스 중앙 저장소에는 단 2개의 Branch만을 유지한다. **master** Branch와 **develop** Branch이다.
master Branch는 **현재 사용자들에게 배포되고 있는 버전**을 관리한다.
그리고 develop Branch는 **현재 개발 중인 버전**을 관리한다.

master Branch는 develop Branch에서 개발을 거친 후 배포를 결정하였을 때 한 번씩 업데이트되며, 버전 별 마지막 Commit에 Tag를 부여한다. 이것은 추후에 특정 개발 버전에서의 소스 코드를 쉽게 찾을 수 있게 하기 위함이다. Tag의 이름 규칙은 **`REL-X.Y.Z`**의 형식을 따른다. X에는 현재 버전 코드가 들어가며, Y에는 주 버전이 유지된 상태로 몇 번의 기능 업데이트가 추가되었는지를 넣고, Z에는 소소한 기능 수정이나 버그 픽스의 횟수를 넣는다.

위 2개의 Branch를 제외한 다른 Branch들은 Pull Request를 위한 유지와 같은 임시적 목적 이외에 다른 목적으로 소스 저장소에 유지하지 않는다.

### 개발자의 PC

개발자의 PC에는 총 4개의 Branch를 관리한다. **release**, **hotfix**, **feature**, **issue**가 그것이다.
이 Branch들은 만들어지고 나서 각 Branch의 목적에 해당하는 Branch(master, develop)에 Pull Request를 날리게 되고, 해당 Pull Request가 승인되면 소멸한다.

#### release

release Branch는 develop Branch로부터 생성한다. 이 Branch가 만들어졌다는 것은 **출시를 목표로 한 버전에 필요한 모든 기능의 개발이 끝났다는 의미**이다.
이 Branch가 만들어진 이후로는 오직 버그 수정 Commit만을 이 Branch에 반영한다.

모든 버그가 픽스된 뒤에 배포가 최종 확정되면 master Branch와 develop Branch에 동시에 Merge한다. 이 작업은 **신규 버전 배포**를 의미하는 것으로, Merge가 완료된 후에는 master Branch의 가장 끝에 존재하는 Commit에 위에 서술한 이름 규칙에 맞는 Tag를 부여한다.

**`release/X.Y.Z`**의 이름 규칙을 따른다. **X.Y.Z**에 들어가는 내용은 Tag의 이름 규칙을 따른다.

#### hotfix

hotfix Branch는 master Branch로부터 생성한다. 이 Branch는 만약 release Branch에서의 작업이 끝나서 master Branch에 Merge를 한 상태고 **실제로 사용자들에게 배포된 버전에서 오류가 발생하였을 때 이를 긴급히 패치하기 위하여** 만들어진다.

해당 버그가 픽스되면 master, develop Branch에 Merge한다. master Branch에 직접 Merge했으므로 Tag를 달아주어야 하는데, 이 경우는 **기존 최신 Tag의 Z값을 1 올려서 지정**한다. 예를 들어 기존 master Branch의 최신 버전이 `REL-3.0.0`이었다면, hotfix Branch를 통해 Merge된 버전은 `REL-3.0.1`의 Tag를 가진다.

**`hotfix/[Issue Number]`**의 이름 규칙을 가진다. [Issue Number] 부분에는 위의 '상황' 단락에서 전제했던 Issue Number를 넣는다.

#### feature

feature Branch는 develop Branch로부터 생성한다. 이 Branch는 정해진 개발 계획에 의해 **새로운 기능을 추가할 때** 만들어진다.

해당 기능이 완성되면 develop Branch에 Merge한다.

**`feature/[Issue Number]/[짧은 설명]`**의 이름 규칙을 가진다. 여기서 [짧은 설명]은, 알파벳 소문자와 하이픈(-)만을 사용하여 간략히 남긴다. 예를 들면, **`feature/237/add-verifications`**와 같이 핵심적인 주제만 간략하게 남긴다.

#### issue

issue Branch는 develop, feature, release Branch 중 하나로부터 생성된다. 이 Branch가 가지는 의미는 **특정 Issue로 지정된 문제 혹은 기능에 대하여 이를 보수하거나 수정하는 작업**이다. 버그 픽스와, Feature를 구현하였는데 일부가 덜 구현되었거나 잘못 구현되어 이를 보충해야 하는 경우 등이 이에 속한다.

해당 작업이 완료되면 생성된 부모 Branch에 Merge한다.

**`issue/[Issue Number]`**의 이름 규칙을 가진다.

### 코드 리뷰

사내 정책으로써, Pull Request 단계에서 코드 리뷰를 통과해야만 Merge가 가능하도록 하였다.
만들어진 Pull Request는 **최소 2명의 Reviewer**가 지정되어야 한다.
그 이후, Reviewer들은 팀 내의 소통 채널(각 Pull Request에 있는 Comment 기능, Slack과 같은 그룹웨어, 팀원간의 대화)를 통해서 코드를 리뷰한다.

Reviewer는 충분한 리뷰를 거친 후, 이 PR이 모든 요구 사항을 만족하였으며 거절될 특별한 이유가 없는 경우 **Approve** 버튼을 눌러 리뷰를 성공적으로 끝마쳤음을 알린다. **Reviewer가 2명일 경우 2명 모두, 3명 이상일 경우 과반수 이상이 Approve하면 Merge를 승인한 것으로 간주**한다.

만약 Reviewer가 리뷰를 진행하던 중 해당 PR이 승인되어선 안되는 이유를 발견한 경우 팀 내의 소통 채널에 그 내용을 알린다. 해당 내용이 확실히 문제가 될 내용이라는 것이 밝혀지면 **문제를 발견한 Reviewer는 Decline 버튼을 누르고 텍스트로 문제를 충분히 설명하여 작업을 재요청**한다. 이 행동은 취소할 수 없으므로 결정에 유의하여야 한다.

우리 회사의 경우 Bitbucket을 사용하고 있기 때문에 Pull Request에 자체적으로 Approve, Decline 버튼이 생성되나, Github는 이러한 구조가 아니다. Github에서는 Pull Request 하단의 Comment를 이용하는데, **Comment, Approve, Request Changes** 중 하나의 옵션을 달아서 Review Comment를 남길 수 있다. Administrator 권한을 지닌 사람은 [Required Review][required-review] 기능을 설정하여 최소 1명의 Approve가 있어야 Merge를 허용하게 제한할 수 있다. 이 이상의 내용은 글의 범위를 벗어나므로, 링크를 통해 확인하기 바란다.

## 정리

이 회사에서의 첫 Sprint가 절반 정도 지나갔을 때 회사에서 이 Branching Strategy를 나름대로 정의한 후, 곧바로 남은 2일 정도의 시간동안 처리되는 Issue들에 이 방법을 도입하였더니 매우 빠르고 체계적으로 Commit들이 정리되는 것을 몸으로 체감할 수 있었다. 이 방법은 회사에서뿐만 아니라 개인 프로젝트를 할 때 역시 사용하기 정말 좋은 전략이라고 생각한다. 이후에 진행되는 프로젝트들에서도 지속적으로 활용할 수 있을 것 같다.

[miro-community]: http://mirocommunity.readthedocs.io/en/latest/internals/branching-model.html
[blog]: http://rumblefish.tistory.com/m/65
[jira]: http://atlassian.net
[bitbucket]: http://bitbucket.org
[required-review]: https://help.github.com/articles/about-required-reviews-for-pull-requests/
