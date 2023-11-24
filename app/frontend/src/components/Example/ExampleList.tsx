import { Example } from "./Example";

import styles from "./Example.module.css";

export type ExampleModel = {
    text: string;
    value: string;
};

const EXAMPLES: ExampleModel[] = [
    { text: "残業には割増賃金が支払われますか？", value: "残業には割増賃金が支払われますか？" },
    { text: "退職金の支給額はどのように計算されますか？", value: "退職金の支給額はどのように計算されますか？" },
    { text: "副業は可能ですか？", value: "副業は可能ですか？" }
];

interface Props {
    onExampleClicked: (value: string) => void;
}

export const ExampleList = ({ onExampleClicked }: Props) => {
    return (
        <ul className={styles.examplesNavList}>
            {EXAMPLES.map((x, i) => (
                <li key={i}>
                    <Example text={x.text} value={x.value} onClick={onExampleClicked} />
                </li>
            ))}
        </ul>
    );
};
