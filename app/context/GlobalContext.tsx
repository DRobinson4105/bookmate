"use client"
import { createContext, useContext, ReactNode, useState } from 'react';

type GlobalContextType = {
  images: File[];
  setImages: (images: File[]) => void;
  isbns: string[][][];
  setIsbns: (ibns: string[][][]) => void;
  
};

const defaultState: GlobalContextType = {
  images: [],
  setImages: () => {},
  isbns: [],
  setIsbns: () => {}
};

const GlobalContext = createContext<GlobalContextType>(defaultState);

export const useGlobalContext = () => useContext(GlobalContext);

type Props = {
    children: ReactNode;
};

export const GlobalProvider: React.FC<Props> = ({ children }) => {
    const [images, setImages] = useState<File[]>([]);
    const [isbns, setIsbns] = useState<string[][][]>([]);

    return (
        <GlobalContext.Provider value={{ images, setImages, isbns, setIsbns }}>
            {children}
        </GlobalContext.Provider>
    );
};
