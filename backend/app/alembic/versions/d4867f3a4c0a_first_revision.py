"""First revision

Revision ID: d4867f3a4c0a
Revises:
Create Date: 2019-04-17 13:53:32.978401

"""
from sqlalchemy.sql.schema import ForeignKeyConstraint
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "d4867f3a4c0a"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "user",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("full_name", sa.String(), nullable=True),
        sa.Column("email", sa.String(), nullable=True),
        sa.Column("hashed_password", sa.String(), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=True),
        sa.Column("is_superuser", sa.Boolean(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_user_email"), "user", ["email"], unique=True)
    op.create_index(op.f("ix_user_full_name"), "user", ["full_name"], unique=False)
    op.create_index(op.f("ix_user_id"), "user", ["id"], unique=False)
    op.create_table(
        "license",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("name", sa.String(), nullable=False, unique=True),
        sa.Column("external_link", sa.String(), nullable=False),
    )
    op.create_index(op.f("ix_license_id"), "license", ["id"], unique=False)
    op.create_index(op.f("ix_license_name"), "license", ["name"], unique=True)
    op.create_table(
        "provider",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("name", sa.String(100), nullable=False, unique=True),
        sa.Column("url", sa.String(1024), nullable=False),
    )
    op.create_index(op.f("ix_provider_id"), "provider", ["id"], unique=False)
    op.create_index(op.f("ix_provider_name"), "provider", ["name"], unique=True)
    op.create_table(
        "track",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("title", sa.UnicodeText, nullable=False),
        sa.Column("artist", sa.UnicodeText),
        sa.Column("internal_media_url", sa.String(512)),
        sa.Column("external_link", sa.String(1024), nullable=False),
        sa.Column("license_id", sa.Integer),
        sa.Column("provider_id", sa.Integer),
        sa.ForeignKeyConstraint(["license_id"], ["license.id"]),
        sa.ForeignKeyConstraint(["provider_id"], ["provider.id"]),
        sa.UniqueConstraint(
            "title", "artist", "provider_id", name="unique_track_details"
        ),
    )
    op.create_index(op.f("ix_track_id"), "track", ["id"], unique=False)
    op.create_table(
        "embedding_model",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("name", sa.String(200), nullable=False, unique=True),
    )
    op.create_index(op.f("ix_embedding_model_id"), "track", ["id"], unique=False)
    op.create_table(
        "embedding",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("track_id", sa.Integer, nullable=False),
        sa.Column("embedding_model_id", sa.Integer, nullable=False),
        sa.ForeignKeyConstraint(["track_id"], ["track.id"]),
        sa.ForeignKeyConstraint(["embedding_model_id"], ["embedding_model.id"]),
        sa.Column("values", sa.ARRAY(sa.Float(precision=32)), nullable=False),
        sa.UniqueConstraint("track_id", "embedding_model_id", name="unique_embedding"),
    )
    op.create_index(
        op.f("ix_embedding_embedding_model_id"), "embedding", ["embedding_model_id"], unique=False
    )
    op.create_index(
        op.f("ix_embedding_track_id"), "embedding", ["track_id"], unique=True
    )

def downgrade():
    op.drop_index(op.f("ix_license_id"), table_name="license")
    op.drop_index(op.f("ix_license_name"), table_name="license")
    op.drop_table("license")
    op.drop_index(op.f("ix_provider_id"), table_name="provider")
    op.drop_index(op.f("ix_provider_name"), table_name="provider")
    op.drop_table("provider")
    op.drop_index(op.f("ix_track_id"), table_name="track")
    op.drop_table("track")
    op.drop_index(op.f("ix_embedding_model_id"), table_name="embedding_model")
    op.drop_table("embedding_model")
    op.drop_index(op.f("ix_embedding_embedding_model_id"), table_name="embedding")
    op.drop_index(op.f("ix_embedding_track_id"), table_name="embedding")
    op.drop_table("embedding")
    op.drop_index(op.f("ix_user_id"), table_name="user")
    op.drop_index(op.f("ix_user_full_name"), table_name="user")
    op.drop_index(op.f("ix_user_email"), table_name="user")
    op.drop_table("user")
