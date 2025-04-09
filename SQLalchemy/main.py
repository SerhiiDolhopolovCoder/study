from sqlalchemy import create_engine, MetaData
from sqlalchemy import Table, Column, ForeignKey
from sqlalchemy import Integer, String, BigInteger

engine = create_engine('sqlite+pysqlite:///:memory:', echo=True)
metadata = MetaData()


user_table = Table(
    'users',
    metadata,
    Column('id', Integer, primary_key=True),
    Column('user_id', BigInteger, unique=True),
    Column('secondname', String(60)),
)

address = Table(
    'addresses',
    metadata,
    Column('id', Integer, primary_key=True),
    Column('user_id', ForeignKey('users.user_id')),
    Column('email', String(60), nullable=False),
)

print(user_table.columns.keys())