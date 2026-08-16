"""create inference_log

Revision ID: 0001
Revises:
Create Date: 2026-08-16
"""

from alembic import op
import sqlalchemy as sa

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "inference_log",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("transaction_id", sa.String(128), nullable=False),
        sa.Column("event_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("customer_id", sa.String(128), nullable=False),
        sa.Column("merchant_id", sa.String(128), nullable=False),
        sa.Column("amount", sa.Float(), nullable=False),
        sa.Column("fraud_probability", sa.Float(), nullable=False),
        sa.Column("decision", sa.String(32), nullable=False),
        sa.Column("risk_level", sa.String(32), nullable=False),
        sa.Column("model_version", sa.String(128), nullable=False),
        sa.Column("threshold", sa.Float(), nullable=False),
        sa.Column("latency_ms", sa.Float(), nullable=False),
        sa.Column("features_json", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_inference_log_transaction_id", "inference_log", ["transaction_id"])
    op.create_index("ix_inference_log_customer_id", "inference_log", ["customer_id"])
    op.create_index("ix_inference_log_decision", "inference_log", ["decision"])


def downgrade() -> None:
    op.drop_table("inference_log")
