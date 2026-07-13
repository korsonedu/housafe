# Task 4 Report: 家庭 / 老人 / 紧急联系人（families）

## TDD Evidence

**RED** (before impl):
```
families/tests/test_families.py::test_create_family_and_list_own_only FAILED [assert 404 == 201]
families/tests/test_families.py::test_family_requires_auth FAILED [assert 404 == 401]
2 failed in 0.54s
```

**GREEN** (after impl):
```
..                                                                       [100%]
2 passed in 0.57s
```

## Files Changed

| File | Action |
|------|--------|
| `backend/families/__init__.py` | Create (empty) |
| `backend/families/apps.py` | Create (name = "families") |
| `backend/families/models.py` | Create (Family, Membership, Elder, Contact) |
| `backend/families/serializers.py` | Create (FamilySerializer, ElderSerializer, ContactSerializer) |
| `backend/families/views.py` | Create (FamilyViewSet, _ChildViewSet, ElderViewSet, ContactViewSet) |
| `backend/families/urls.py` | Create (4 routes under /api/) |
| `backend/families/tests/__init__.py` | Create (empty) |
| `backend/families/tests/test_families.py` | Create (2 tests) |
| `backend/housafe/settings.py` | Modify (INSTALLED_APPS add "families") |
| `backend/housafe/urls.py` | Modify (add path("api/", include("families.urls"))) |

## Self-Review Findings

1. **Per-owner scoping correct**: `FamilyViewSet.get_queryset()` filters by `owner=self.request.user`; `_ChildViewSet._family()` uses `owner=self.request.user` too, so elders/contacts are scoped through the family owner.
2. **Module isolation**: families only uses `request.user` from accounts -- no cross-app imports.
3. **Membership model created but no REST endpoints** -- per plan spec, that's fine.
4. **PostgreSQL unavailable** during test; used `DATABASE_URL=sqlite:///db.sqlite3` fallback per brief.
5. **Contact ordering**: `Contact.Meta.ordering = ["order"]` ensures contacts are returned in correct order by default.

## Commit

`feat: families/elders/contacts with per-owner scoping`
