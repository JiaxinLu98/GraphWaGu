import { default as React } from 'react';
type TutorialProps = {
    unmount: () => void;
};
declare class Tutorial extends React.Component<TutorialProps, {}> {
    constructor(props: TutorialProps | Readonly<TutorialProps>);
    handleSubmit(event: {
        preventDefault: () => void;
    }): void;
    render(): import("react/jsx-runtime").JSX.Element;
}
export default Tutorial;
