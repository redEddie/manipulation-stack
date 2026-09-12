"""삭제 대기 목록 -- 되돌릴 수 없는 조작으로 가는 **유일한 문**.

**왜 있는가.** 2026-09-10 조사에서 에피소드 삭제로 가는 문이 넷이었다:
왼쪽 패널 빨간 버튼, Dataset 메뉴, Analysis 순위표 버튼, 그리고 파일 통째
삭제. 실행 함수는 하나로 모여 있었지만(``delete_episodes``) 누르는 자리가
넷이었고, 그중 둘은 "선택" 바로 옆에 있었다. 실제로 오클릭 사고가 나서 파일
삭제를 메뉴로 옮긴 이력도 있다.

**규칙 하나로 정리한다: 되돌릴 수 있는 것은 즉시, 되돌릴 수 없는 것은 여기를
거친다.** 재판정(성공↔실패)은 뒤집으면 복구되므로 즉시 실행이고, 삭제는
표시만 해 두었다가 한 번에 실행한다.

그래서 **발견하는 곳은 여럿, 실행하는 문은 하나**다. 격자·트리·순위표
어디서든 표시할 수 있고, 지우는 것은 "삭제 실행" 하나뿐이다. 표시는 취소가
자유로우니 겁 없이 누를 수 있고, 확인창은 배치 전체를 한 번에 보여준다 --
검토가 가장 값진 순간이 바로 그때다.

**한 번에 지우면 renumber 도 한 번이다.** 삭제마다 uid 가 재배정되어 파생
캐시(썸네일·프록시 클립)를 통째로 버려야 하는데, 열 번 지우면 열 번 다시
굽는다. 모아서 한 번 지우면 한 번이다.

파일 통째 삭제는 여기 들어오지 않는다 -- 그것은 태스크 하나가 날아가는
다른 무게의 조작이라 메뉴에 따로 둔다.
"""

from __future__ import annotations

from pathlib import Path


class CurationBasket:
    """``(파일경로, 에피소드이름)`` 의 순서 있는 집합.

    창이 하나 갖고 있고 화면들이 함께 쓴다. 위젯을 모르므로 하드웨어도 화면도
    없이 시험된다.
    """

    def __init__(self) -> None:
        self._items: list = []          # 순서 유지 -- 표시한 순서로 보여준다

    # ------------------------------------------------------------------ 조회
    def __len__(self) -> int:
        return len(self._items)

    def __contains__(self, key) -> bool:
        return self._norm(key) in self._items

    def __iter__(self):
        return iter(self._items)

    @staticmethod
    def _norm(key):
        path, name = key
        return (str(Path(path)), str(name))

    def items(self) -> list:
        return list(self._items)

    def by_file(self) -> dict:
        """``delete_episodes`` 가 받는 모양. 파일별로 묶는다.

        한 파일 안에서 하나씩 지우고 매번 번호를 다시 매기면 두 번째부터는
        이미 밀린 이름을 지우게 된다 -- 그래서 실행부가 파일별 묶음을 받는다.
        """
        out: dict = {}
        for path, name in self._items:
            out.setdefault(Path(path), []).append(name)
        return out

    def count_for(self, path) -> int:
        """이 파일에서 표시된 개수. 씬을 옮겨 다닐 때 화면에 쓴다."""
        p = str(Path(path))
        return sum(1 for q, _ in self._items if q == p)

    # ------------------------------------------------------------------ 변경
    def add(self, key) -> bool:
        k = self._norm(key)
        if k in self._items:
            return False
        self._items.append(k)
        return True

    def discard(self, key) -> bool:
        k = self._norm(key)
        if k not in self._items:
            return False
        self._items.remove(k)
        return True

    def toggle(self, key) -> bool:
        """표시하면 True, 표시를 풀면 False.

        화면은 add/discard 를 따로 부른다(격자는 표시, 목록은 해제) -- 지금
        이 메서드의 소비자는 계약 테스트뿐이지만, 한 번 누를 때마다 뒤집는
        타일을 만들면 곧 쓰인다. 남겨 두는 근거를 여기 적어 둔다.
        """
        if self.discard(key):
            return False
        self.add(key)
        return True

    def clear(self) -> None:
        self._items.clear()

    def drop_file(self, path) -> int:
        """한 파일의 표시를 전부 지운다. 반환: 지워진 개수.

        삭제를 실행한 뒤에 부른다 -- 실행하지 않은 표시가 남아 있으면 다음
        실행이 이미 없는 에피소드를 지우려 든다.
        """
        p = str(Path(path))
        before = len(self._items)
        self._items = [(q, n) for q, n in self._items if q != p]
        return before - len(self._items)
