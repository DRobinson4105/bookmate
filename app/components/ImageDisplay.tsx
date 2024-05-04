import React, { useState, useEffect } from 'react';

interface Box {
  x: number;
  y: number;
  width: number;
  height: number;
  isbn: string;
}

interface BoundingBoxProps {
  box: Box;
  isbn: string;
  removeBox: (isbn: string) => void;
}

const BoundingBox: React.FC<BoundingBoxProps> = ({ box, isbn, removeBox }) => {
  const [hovered, setHovered] = useState(false);

  const handleRemove = () => {
    removeBox(isbn);
  };

  return (
    <div
      style={{
        position: 'absolute',
        top: box.y,
        left: box.x,
        width: box.width,
        height: box.height,
        border: '2px solid red',
        pointerEvents: 'auto',
        cursor: 'pointer',
      }}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
    >
      {hovered && (
        <button
          style={{
            position: 'absolute',
            top: '-15px',
            right: '-15px',
            background: 'none',
            border: 'none',
            color: 'red',
            cursor: 'pointer',
          }}
          onClick={handleRemove}
        >
          X
        </button>
      )}
      <div
        style={{
          position: 'absolute',
          top: '-25px',
          left: '50%',
          transform: 'translateX(-50%)',
          background: 'white',
          padding: '5px',
          borderRadius: '5px',
          boxShadow: '0 0 5px rgba(0,0,0,0.5)',
        }}
      >
        {isbn}
      </div>
    </div>
  );
};

interface BoundingBoxesProps {
  boxes: Box[];
}

const BoundingBoxes: React.FC<BoundingBoxesProps> = ({ boxes }) => {
  const [isbnList, setIsbnList] = useState(boxes.map(box => box.isbn));

  const removeBox = (isbnToRemove: string) => {
    setIsbnList(prevIsbnList => prevIsbnList.filter(isbn => isbn !== isbnToRemove));
  };

  return (
    <div style={{ position: 'relative' }}>
      {boxes.map(box => (
        <BoundingBox
          key={box.isbn}
          box={box}
          isbn={box.isbn}
          removeBox={removeBox}
        />
      ))}
    </div>
  );
};

const BoundingBoxesComponent: React.FC<{ image: File }> = ({ image }) => {
  const [imageSize, setImageSize] = useState<{ width: number; height: number }>({ width: 0, height: 0 });
  const [boxes, setBoxes] = useState<Box[]>([
    { x: -50, y: -100, width: 100, height: 100, isbn: '1234567890' },
    { x: 200, y: 200, width: 150, height: 150, isbn: '0987654321' },
  ]);

  useEffect(() => {
    const reader = new FileReader();
    reader.onload = (event) => {
      const img = new Image();
      img.onload = () => {
        setImageSize({ width: img.width, height: img.height });
      };
      img.src = event.target?.result as string;
    };
    reader.readAsDataURL(image);
  }, [image]);

  return (
    <div>
      {imageSize.width !== 0 && (
        <div style={{ position: 'relative' }}>
          <img src={URL.createObjectURL(image)} alt="Your Image" style={{ width: imageSize.width / 10, height: imageSize.height / 10 }} />
          <BoundingBoxes boxes={boxes} />
        </div>
      )}
    </div>
  );
};

export default BoundingBoxesComponent;
