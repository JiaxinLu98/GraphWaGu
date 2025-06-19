export declare class Controller {
    mousemove: ((prevMouse: number[], curMouse: number[], evt: MouseEvent) => void) | null;
    press: ((curMouse: number[], evt: MouseEvent) => void) | null;
    wheel: ((amount: number) => void) | null;
    constructor();
    registerForCanvas(canvas: HTMLCanvasElement): void;
}
