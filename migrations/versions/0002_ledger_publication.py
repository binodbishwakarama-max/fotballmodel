"""Track immutable public ledger and OpenTimestamps publication references."""

import sqlalchemy as sa
from alembic import op

revision = "0002_ledger_publication"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    columns = {column["name"] for column in sa.inspect(op.get_bind()).get_columns("ledger_batches")}
    if "timestamp_proof_commit" not in columns:
        op.add_column(
            "ledger_batches",
            sa.Column("timestamp_proof_commit", sa.String(length=64), nullable=True),
        )


def downgrade() -> None:
    columns = {column["name"] for column in sa.inspect(op.get_bind()).get_columns("ledger_batches")}
    if "timestamp_proof_commit" in columns:
        op.drop_column("ledger_batches", "timestamp_proof_commit")
