from sqlalchemy import create_engine
from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import Session
from sqlalchemy.orm import Mapped, mapped_column, as_declarative
from sqlalchemy.orm import relationship


engine = create_engine("sqlite:///test.db", echo=True)

@as_declarative()
class Base(): pass


class User(Base):
    __tablename__ = 'users'
    
    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(50), nullable=False)
    age: Mapped[int] = mapped_column(nullable=False)
    company_id: Mapped[int] = mapped_column(ForeignKey('companies.id'), nullable=True)
    company = relationship("Company", back_populates="users")
    
class Company(Base):
    __tablename__ = 'companies'
    
    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(50), nullable=False)
    users = relationship("User", back_populates="company")
    
Base.metadata.create_all(engine)


with Session(engine) as session:
    user = User(name="John Doe", age=30)
    session.add(user)
    session.commit()