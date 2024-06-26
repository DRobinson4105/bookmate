"use client"
import { createContext, useContext, ReactNode, useState } from 'react';

type GlobalContextType = {
  images: File[];
  setImages: (images: File[]) => void;
  boxedImages: string[];
  setBoxedImages: (ibns: string[]) => void;
  isbns: string[];
  setIsbns: (ibns: string[]) => void;
  spreadsheet: string;
  setSpreadsheet: (images: string) => void;
};

const defaultState: GlobalContextType = {
  images: [],
  setImages: () => {},
  boxedImages: [],
  setBoxedImages: () => {},
  isbns: [],
  setIsbns: () => {},
  spreadsheet: '',
  setSpreadsheet: () => {},
};

const GlobalContext = createContext<GlobalContextType>(defaultState);

export const useGlobalContext = () => useContext(GlobalContext);

type Props = {
    children: ReactNode;
};

export const GlobalProvider: React.FC<Props> = ({ children }) => {
    const [images, setImages] = useState<File[]>([]);
    const [boxedImages, setBoxedImages] = useState<string[]>([]);
    const [isbns, setIsbns] = useState<string[]>([]);
    const [spreadsheet, setSpreadsheet] = useState<string>('');

    return (
        <GlobalContext.Provider value={{ images, setImages, boxedImages, setBoxedImages, isbns, setIsbns, spreadsheet, setSpreadsheet }}>
            {children}
        </GlobalContext.Provider>
    );
};
