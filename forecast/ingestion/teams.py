"""Team name normalization and source-alias resolution."""

from __future__ import annotations

import re
import unicodedata

from sqlalchemy import select
from sqlalchemy.orm import Session

from forecast.models import Team, TeamAlias


def normalize_team_name(name: str) -> str:
    """Return a stable case-insensitive key that ignores accents and punctuation."""
    decomposed = unicodedata.normalize("NFKD", name)
    ascii_name = "".join(char for char in decomposed if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", " ", ascii_name.casefold()).strip()


def resolve_team(
    session: Session,
    *,
    source: str,
    raw_name: str,
    league: str,
    allow_primary_bootstrap: bool = False,
) -> Team:
    """Resolve a source name or bootstrap a canonical alias for the primary source."""
    normalized = normalize_team_name(raw_name)
    alias = session.scalar(
        select(TeamAlias).where(
            TeamAlias.source == source,
            TeamAlias.normalized_name == normalized,
        )
    )
    if alias is not None:
        return alias.team

    canonical_match = session.scalar(
        select(Team).where(
            Team.league == league,
            Team.canonical_name == raw_name.strip(),
        )
    )
    if canonical_match is not None and not allow_primary_bootstrap:
        session.add(
            TeamAlias(
                team_id=canonical_match.id,
                source=source,
                raw_name=raw_name.strip(),
                normalized_name=normalized,
            )
        )
        session.flush()
        return canonical_match

    if allow_primary_bootstrap:
        canonical_key = normalize_team_name(raw_name)
        team = session.scalar(
            select(Team).where(
                Team.league == league,
                Team.canonical_name == raw_name.strip(),
            )
        )
        if team is None:
            team = Team(canonical_name=raw_name.strip(), league=league)
            session.add(team)
            session.flush()
        session.add(
            TeamAlias(
                team_id=team.id,
                source=source,
                raw_name=raw_name.strip(),
                normalized_name=canonical_key,
            )
        )
        session.flush()
        return team

    raise ValueError(f"Unknown team name from {source!r}: {raw_name!r}")
