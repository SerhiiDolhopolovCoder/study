from sqlalchemy import create_engine, MetaData
from sqlalchemy import Table, ForeignKey
from sqlalchemy import Integer, String, BigInteger
from sqlalchemy.orm import as_declarative, declared_attr
from sqlalchemy.orm import mapped_column, Mapped, Session


engine = create_engine('sqlite+pysqlite:///:memory:', echo=True)


@as_declarative()
class AbstractModel():
    id: Mapped[int] = mapped_column(autoincrement=True, primary_key=True)
    
    @classmethod
    @declared_attr
    def __table_name__(cls) -> str:
        return cls.__name__.lower()


class UserModel(AbstractModel):
    __tablename__ = 'users'
    name: Mapped[str] = mapped_column(nullable=False)
    fullname: Mapped[str] = mapped_column()
    
    
class AddressModel(AbstractModel):
    __tablename__ = 'addresses'
    email: Mapped[str] = mapped_column(nullable=False)
    user_id: Mapped[int] = mapped_column(ForeignKey('{0}.id'.format(UserModel.__tablename__)))

    
with Session(engine) as session:
    with session.begin():
        print('{0}.id'.format(UserModel.__tablename__))
        AbstractModel.metadata.create_all(engine) 
        user = UserModel(name='John', fullname='John Doe')
        session.add(user)
        user = UserModel(name='Johnfgfgfgffgfgfgff', fullname='John Doe')
        session.add(user)
        
    with session.begin():
        user = session.scalar(UserModel.__table__.select().where(UserModel.name == 'John'))
        print('user:', user)