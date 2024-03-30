import {
    Autocomplete,
    Checkbox,
    FormGroup,
    InputLabel,
    Paper,
    Select,
    Slider, TextField,
    Typography
} from "@mui/material";
import Button from "@mui/material/Button";

import {useRecoilState, useRecoilValue} from "recoil";
import {pathsState, settingsState} from "./atoms.ts";
import Container from "@mui/material/Container";
import {useRef} from "react";


export const RequestForm = () => {
    const [settings, setSettings] = useRecoilState(settingsState);
    const paths = useRecoilValue(pathsState);
    const pathOptions = paths.filter((path) => path.path.length == settings.boardSize * settings.boardSize);
    const autocompleteRef = useRef<typeof Autocomplete>(null);
    console.log(pathOptions);
    return (
        <Paper
            style={{
                padding: 20,
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                justifyContent: "center",
                width: "50%",
                margin: "auto",
                marginTop: 20,

            }
            }
        >

            <FormGroup
                style={{
                    padding: 20,
                    width: "100%",
                }}

            >
                <h2>Board Generation</h2>
                <Typography id="input-slider-size" gutterBottom>
                    Board size
                </Typography>
                <Slider
                    defaultValue={8}
                    min={2}
                    max={16}
                    value={
                        settings.boardSize
                    }
                    onChange={
                        (_, v) => {
                            setSettings({...settings, boardSize: v as number})
                        }
                    }
                    marks
                    step={1}
                    aria-labelledby={"input-slider-size"}
                    valueLabelDisplay="on"


                />
                <Typography id="input-slider-colors" gutterBottom>
                    Number of Colors
                </Typography>
                <Slider
                    defaultValue={12}
                    min={4}
                    max={22}
                    value={
                        settings.boardColors
                    }
                    onChange={
                        (_, v) => setSettings({...settings, boardColors: v as number})
                    }
                    marks
                    step={1}
                    aria-labelledby={"input-slider-colors"}
                    valueLabelDisplay="on"
                />


            </FormGroup>
            <Button type="submit" color="primary">Generate</Button>

            <FormGroup
                style={{
                    padding: 20,
                    width: "100%",
                }}
            >
                <h2>Solver</h2>
                <FormGroup>
                    <Autocomplete
                        id="paths"
                        renderInput={(params) => <TextField
                            {...params}
                            variant="standard"
                            label="Paths"
                            placeholder="Paths"
                        />}
                        options={pathOptions}
                        ref={autocompleteRef}


                    >
                    </Autocomplete>
                </FormGroup>

            </FormGroup>
            <FormGroup
                style={{
                    padding: 20,
                    width: "100%",
                    display: "flex",
                    alignItems: "start",
                }}
            >
                <h2>Options</h2>
                <Container
                    style={{
                        display: "flex",
                        flexDirection: "row",
                        alignItems: "center",
                    }}
                ><Typography
                    id={"use-cache"}
                >
                    Use Cache
                </Typography>

                    <Checkbox

                        color="primary"
                        aria-labelledby={"use-cache"}
                        style={{
                            padding: 20,
                        }}
                    /></Container>
            </FormGroup>

            <Button type="submit" color="primary">Solve</Button>
            <Button type="submit" color="primary">Step By Step</Button>
        </Paper>
    )
}
