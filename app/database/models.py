from sqlalchemy import create_engine, ForeignKey, Boolean, Column, Integer, String, DateTime, Float
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import relationship, declarative_base, sessionmaker
from database.database import Base
import uuid

class Image(Base):
    __tablename__ = 'image'
    __table_args__ = {'extend_existing': True}
    
    id = Column('Id_Image', UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    url = Column('URL', String)
    description = Column('Description', String)
    meta_data = Column('Meta_data', JSONB, nullable=True)
    metric = Column('Metric', JSONB, nullable=True)

    annotations = relationship('Annotation', back_populates='image')  # Quan hệ với bảng annotations
    imagedatasets = relationship('ImageDataset', back_populates='image')

class Annotation(Base):  
    __tablename__ = 'annotations'  
    __table_args__ = {'extend_existing': True}
    
    id = Column('Id_Annotation', UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)  # Đổi tên cột
    image_id = Column('Image_Id', UUID(as_uuid=True), ForeignKey('image.Id_Image'), nullable=False)
    label_id = Column('Label_Id', UUID(as_uuid=True), ForeignKey('label.Id_Label'), nullable=False)
    segment = Column('Segment', JSONB, nullable=True)
    bbox = Column('Bbox', JSONB, nullable=True)
    motion_mask = Column('Motion_Mask', JSONB, nullable=True)

    image = relationship('Image', back_populates='annotations')  # Quan hệ với bảng Image
    label = relationship('Label', back_populates='annotations')  # Quan hệ với bảng Label

class Dataset(Base):
    __tablename__ = 'dataset'
    __table_args__ = {'extend_existing': True}
    
    id = Column('Id_Dataset', UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column('Name', String, nullable=False, unique=True)
    description = Column('Description', String, nullable=True)
    type = Column('Type', String, nullable=False)
    version = Column('Version', String, nullable=False)

    imagedatasets = relationship('ImageDataset', back_populates='dataset')
    datasetversions = relationship('DatasetVersion', back_populates='dataset')
    labeldatasets = relationship('LabelDataset', back_populates='dataset')

class ImageDataset(Base):
    __tablename__ = 'image_dataset'
    __table_args__ = {'extend_existing': True}
    
    id = Column('Id_ImageDataset', UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    image_id = Column('Image_Id', UUID(as_uuid=True), ForeignKey('image.Id_Image'), nullable=False)
    dataset_id = Column('Dataset_Id', UUID(as_uuid=True), ForeignKey('dataset.Id_Dataset'), nullable=False)

    image = relationship('Image', back_populates='imagedatasets')
    dataset = relationship('Dataset', back_populates='imagedatasets')

class Label(Base):
    __tablename__ = 'label'
    __table_args__ = {'extend_existing': True}
    
    id = Column('Id_Label', UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    class_name = Column('Class_Name', String, nullable=False)

    labeldatasets = relationship('LabelDataset', back_populates='label')
    annotations = relationship('Annotation', back_populates='label')  # Quan hệ 1-N với Annotation

class LabelDataset(Base):
    __tablename__ = 'label_dataset'
    __table_args__ = {'extend_existing': True}
    
    id = Column('Id_LabelDataset', UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    label_id = Column('Id_Label', UUID(as_uuid=True), ForeignKey('label.Id_Label'), nullable=False)
    dataset_id = Column('Dataset_Id', UUID(as_uuid=True), ForeignKey('dataset.Id_Dataset'), nullable=False)

    label = relationship('Label', back_populates='labeldatasets')
    dataset = relationship('Dataset', back_populates='labeldatasets')

class DatasetVersion(Base):
    __tablename__ = 'dataset_version'
    __table_args__ = {'extend_existing': True}
    
    id = Column('Id_DatasetVersion', UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    dataset_id = Column('Id_Dataset', UUID(as_uuid=True), ForeignKey('dataset.Id_Dataset'), nullable=False)
    version_id = Column('Id_Version', UUID(as_uuid=True), ForeignKey('version.Id_Version'), nullable=False)

    dataset = relationship('Dataset', back_populates='datasetversions')
    version = relationship('Version', back_populates='datasetversions')

class Version(Base):
    __tablename__ = 'version'
    __table_args__ = {'extend_existing': True}
    
    id = Column('Id_Version', UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    version_number = Column('Version_Number', String, nullable=False)

    datasetversions = relationship('DatasetVersion', back_populates='version')
