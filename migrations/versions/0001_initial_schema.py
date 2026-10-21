"""Create the initial forecast database schema."""

from alembic import op

from forecast import models  # noqa: F401
from forecast.db import Base

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    Base.metadata.create_all(bind=op.get_bind())
    dialect = op.get_bind().dialect.name
    if dialect == "postgresql":
        op.execute(
            """
            CREATE FUNCTION reject_prediction_mutation()
            RETURNS trigger AS $$
            BEGIN
                RAISE EXCEPTION 'predictions are append-only';
            END;
            $$ LANGUAGE plpgsql
            """
        )
        op.execute(
            """
            CREATE TRIGGER predictions_no_mutation
            BEFORE UPDATE OR DELETE ON predictions
            FOR EACH ROW EXECUTE FUNCTION reject_prediction_mutation()
            """
        )


def downgrade() -> None:
    Base.metadata.drop_all(bind=op.get_bind())
